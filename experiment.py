import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.step_counter = 0 
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        if self.params['auto_sample'] and self.step_counter % self.params['sample_every_n_steps'] == 0:
            self.sample_save_images(batch)

        real_img, labels, top_noise = batch['IMG'], batch['FRAME'], batch['TOP_TRAIN_NOISE']
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels, top_noise = top_noise)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        self.step_counter += 1

        return train_loss['loss']
    
    def sample_save_images(self, batch):
        # Get sample reconstruction image            
        test_input, test_label, test_top_noise = batch['IMG'], batch['FRAME'], batch['TOP_TRAIN_NOISE']
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label, top_noise = test_top_noise)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.step_counter}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(test_input.shape[0],
                                        self.curr_device,
                                        labels = test_label,
                                        top_noise = test_top_noise)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.step_counter}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        
    def on_validation_end(self) -> None:
        self.sample_images()
    
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(test_input.shape[0],
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        ########### similar with diffusion training ##########         
        optimizer = optim.Adam([{'params':self.model.parameters(),'initial_lr':self.params['LR']}], lr=self.params['LR'])
        optims.append(optimizer)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5,verbose = True) # lr = lr*gamma, every 20 epochs Initial_lr = 2e-4
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50,70,90], last_epoch= -1,gamma=0.5)# ori scheduler
        scheds.append(scheduler)
        return optims, scheds

        ########## origin #######
        # optimizer = optim.Adam(self.model.parameters(),
        #                        lr=self.params['LR'],
        #                        weight_decay=self.params['weight_decay'])
        # optims.append(optimizer)
        # # Check if more than 1 optimizer is required (Used for adversarial training)
        # try:
        #     if self.params['LR_2'] is not None:
        #         optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
        #                                 lr=self.params['LR_2'])
        #         optims.append(optimizer2)
        # except:
        #     pass

        # try:
        #     if self.params['scheduler_gamma'] is not None:
        #         scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
        #                                                      gamma = self.params['scheduler_gamma'], verbose=True)
        #         scheds.append(scheduler)

        #         # Check if another scheduler is required for the second optimizer
        #         try:
        #             if self.params['scheduler_gamma_2'] is not None:
        #                 scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
        #                                                               gamma = self.params['scheduler_gamma_2'], verbose=True)
        #                 scheds.append(scheduler2)
        #         except:
        #             pass
        #         return optims, scheds
        # except:
        #     return optims



       




        
