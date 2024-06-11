import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import math


def idx2onehot(idx, n):
    # idx = idx.squeeze() # idx: size([batchsize, 1])--> size([batchsize])
    idx = idx.long()
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ConditionalVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 img_size:int = 64,
                 **kwargs) -> None:
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_classess = num_classes
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # onehot
        # linear_weight is high
        # self.onehot_embed_class = nn.Linear(num_classes, img_size * img_size)
        
        # nn_embed
        # linear_weight is still high and changes with num_classes/num_images
        # y_nn_embed_dim = latent_dim
        # self.y_nn_embedding_table = nn.Embedding(num_classes, y_nn_embed_dim)
        # nn.init.normal_(self.y_nn_embedding_table.weight, std=0.02)
        # self.nn_embed_class = nn.Linear(y_nn_embed_dim, img_size * img_size)

        # timestep_embed, as DiT
        y_timestep_embed_dim = latent_dim
        self.y_timestep_embed = TimestepEmbedder(y_timestep_embed_dim)
        nn.init.normal_(self.y_timestep_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_timestep_embed.mlp[2].weight, std=0.02)
        self.timestep_embed_class = nn.Linear(y_timestep_embed_dim, img_size * img_size)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
            # hidden_dims = [8, 8, 16, 32, 64]
            # hidden_dims = [16, 32, 64, 128, 256]
        
        self.decode_init_dim = hidden_dims[-1]

        in_channels += 1 # To account for the extra label channel
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_output_dim = int(self.img_size*((1/2)**(len(hidden_dims))))
        self.fc_mu = nn.Linear(hidden_dims[-1]*self.encoder_output_dim*self.encoder_output_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*self.encoder_output_dim*self.encoder_output_dim, latent_dim)


        # Build Decoder
        modules = []
        
        # onehot
        # self.onehot_decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] * self.encoder_output_dim * self.encoder_output_dim)

        # nn_embed
        # self.nnembed_decoder_input = nn.Linear(latent_dim + y_nn_embed_dim, hidden_dims[-1] * self.encoder_output_dim * self.encoder_output_dim)

        # timestep_embed
        self.timestep_embed_decoder_input = nn.Linear(latent_dim + y_timestep_embed_dim, hidden_dims[-1] * self.encoder_output_dim * self.encoder_output_dim)


        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input) # size=([batch_size, hidden_dims[-1], img_size*((1/2)**(1/len(hidden_dims))), img_size*((1/2)**(1/len(hidden_dims)))])
        result = torch.flatten(result, start_dim=1) # size=([batch_size, -1])

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result) # size=([8,128])
        log_var = self.fc_var(result) # size=([8,128])

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        # result = self.onehot_decoder_input(z)
        # result = self.nnembed_decoder_input(z)
        result = self.timestep_embed_decoder_input(z)
        result = result.view(-1, self.decode_init_dim, self.encoder_output_dim, self.encoder_output_dim)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = kwargs['labels'].float()

        # onehot
        # y_include = idx2onehot(y, self.num_classess) # size = ([batch_size, num_classes])
        # embedded_class = self.onehot_embed_class(y_include)  # linear matrix, num_classess * (img_size * img_size)

        # # y_nn_embed
        # y_include = self.y_nn_embedding_table(y.int()) # size = ([batch_size, y_nn_embed_dim])
        # embedded_class = self.nn_embed_class(y_include) # linear matrix, y_nn_embed_dim * (img_size * img_size)
        
        # y_timestep_embed
        y_include = self.y_timestep_embed(y.int()) # size = ([batch_size, y_timestep_embed_dim])
        embedded_class = self.timestep_embed_class(y_include) # linear matrix, y_timestep_embed_dim * (img_size * img_size)
        
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1) # size = ([batch_size, 1, img_size, img_size])
        embedded_input = self.embed_data(input)

        x = torch.cat([embedded_input, embedded_class], dim = 1)
        mu, log_var = self.encode(x) # size=([batch_size, latent_dim])

        z = self.reparameterize(mu, log_var) 

        z = torch.cat([z, y_include], dim = 1)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = kwargs['labels'].float()

        # # onehot
        # y_include = idx2onehot(y, self.num_classess) # size = ([batch_size, num_classes])

        # # nn_embed
        # y_include = self.y_nn_embedding_table(y.int()) # size = ([batch_size, y_nn_embed_dim])

        # timestep_embed
        y_include = self.y_timestep_embed(y.int()) # size = ([batch_size, y_timestep_embed_dim])

        z = torch.randn(num_samples,
                        self.latent_dim)

        # z = z.to(current_device)
        z = z.to(y_include.device)

        z = torch.cat([z, y_include], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]