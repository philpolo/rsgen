#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 15:42:12 2025

@author: polo
"""

import wandb
import torch
import random
import yaml, os, sys
import pickle as pkl
import pandas as pd
from tqdm import tqdm
from time import time
from pathlib import Path
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from diffusers import StableDiffusionPipeline

parent = Path('..').resolve()
sys.path.extend([
    os.path.dirname(sys.path[0]), 
    str(parent), str(parent.parent), 
    os.path.join(parent.parent, 'gbsed')])

from utils.datasetGenerator import ds_gen

class Trainer:
    def __init__(self, config_file):
        with open(config_file, 'rb') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = self.config['model_configuration']['model_name']
        self.checkpoint = self.config['model_configuration']['checkpoint']
        self.sample_size = self.config['model_configuration']['sample_size']
        self.lr = self.config['training_configuration']['lr']
        self.batch_size = self.config['training_configuration']['batch_size']
        self.transform_img = transforms.Compose(
            [
                transforms.Resize(
                    self.sample_size, 
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5], [0.5])
            ]
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
        self.pipe = self.pipe.to(self.device)
        self.model_loaded = False
        self.tokenizer = self.pipe.tokenizer
        self.build_datasets()
        self.text_encoder = self.pipe.text_encoder
        self.scheduler = self.pipe.scheduler
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.lr)        
        self.criterion = F.mse_loss
        self.factor = self.config["training_configuration"]['factor']
        self.output_dir = self.config["output_dir"]
        self.caption_file = os.path.join(self.output_dir, "caption.txt")
        self.best_iter = 0
        self.best_loss = 1e6
    
    def build_datasets(self):
        """
        build_datasets creates and saves new train, test and valid datasets if their don't exist in the folder given in the configuration file. 

        Returns
        -------
        None.

        """
        train_ds_path = self.config['datasets']['train_dataset']
        test_ds_path = self.config['datasets']['test_dataset']
        valid_ds_path = self.config['datasets']['valid_dataset']
        if os.path.exists(train_ds_path) \
            and os.path.exists(test_ds_path) \
            and os.path.exists(valid_ds_path):
            with open(train_ds_path, 'rb') as f:
                self.train_dataset = pkl.load(f)
                
            with open(test_ds_path, 'rb') as f:
                self.test_dataset = pkl.load(f)
                
            with open(valid_ds_path, 'rb') as f:
                self.valid_dataset = pkl.load(f)
        else:
            config_filename = self.config["configuration"]['config_filename']
            input_datasets = self.config["input_datasets"]
            gen = ds_gen(
                config_filename, 
                input_datasets, 
                self.transform_img, 
                self.tokenizer
            )
            gen.create_datasets()
            gen.save_datasets(train_ds_path, test_ds_path, valid_ds_path)
            self.train_dataset = gen.train_dataset
            self.test_dataset = gen.test_dataset
            self.valid_dataset = gen.valid_dataset
            
    def save_model(self):
        """
        Saves the model state, the best iteration, the best loss and the optimizer state.

        Returns
        -------
        None.

        """
        torch.save(
            {
                "best_iter":self.best_iter, 
                "best_loss":self.best_loss, 
                "unet_state": self.unet.state_dict(), 
                "optimizer" : self.optimizer.state_dict()
            }, 
            self.checkpoint
        )
        print("Checkpoint saved !")
    
    def load_model(self, training=True):
        """
        If the model checkpoint exists, load it and set the model_loaded flag to true.

        Parameters
        ----------
        training : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        if os.path.exists(self.checkpoint):
            checkpoint = torch.load(self.checkpoint)
            self.unet.load_state_dict(checkpoint["unet_state"])
            if training: 
                self.best_iter = checkpoint["best_iter"]
                self.best_loss = checkpoint['best_loss']
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("Checkpoint loaded !")
            self.model_loaded = True
    
    def test_show(self):
        """
        Selects randomly five images in the test_dataset, Generates five images corresponding to the initial images. 
        Save all the images in the output directory.

        Returns
        -------
        None.

        """
        n = len(self.test_dataset)
        lines = []
        durations = []
        for i in range(5):
            idx = random.randint(0, n - 1)
            caption, img, _, _, _ = self.test_dataset[idx]
            for j in range(5):
                image, duration = self.inference(caption)
                image.save(self.output_dir + f"img_{i}_pred_{j}.png")
                durations.append(duration)
            img = ((img.clamp(-1, 1) + 1) / 2)
            save_image(img, self.output_dir + f"gt_{i}.png")
            lines.append(f'{i}:{caption}\n')
        with open(self.caption_file, 'w') as f:
            f.writelines(lines)       
        df = pd.DataFrame({"duration": durations})
        df.to_csv(os.path.join(self.output_dir, "times.csv"), index=False)
                
    
    def train(self):
        """
        Fine-tunes the stable diffusion training to generate road scene images.

        Returns
        -------
        None.

        """
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        self.load_model()
        self.run = wandb.init(
            project=self.config["wandb_configuration"]['project'], 
            entity=self.config['wandb_configuration']['entity']
        )
        tqdm_bar = tqdm(
            range(
                self.best_iter, 
                self.config['training_configuration']['num_epochs']
            ), desc='Training'
        )
        self.test_show()
        for epoch_idx in tqdm_bar:
            self.unet.train()
            epoch_loss, car_loss, img_loss = 0.0, 0.0, 0.0
            for i, batch in enumerate(train_loader):
                _, images, boxes, input_ids, attention_mask = batch
                images = images.to(self.device)
                boxes = boxes.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                self.optimizer.zero_grad()
                
                with torch.no_grad():
                    latents = self.vae.encode(images).latent_dist.sample() \
                        * self.vae.config.scaling_factor
                    encoder_hidden_states = self.text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    ).last_hidden_state
            
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    self.scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device
                ).long()
                noisy_latents = self.scheduler.add_noise(
                    latents, 
                    noise, 
                    timesteps
                )
                
                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states
                ).sample
                
                interpolate_boxes = F.interpolate(
                    boxes.to(latents.device),
                    size=latents.shape[-2:],
                    mode="nearest"
                )
                
                all_image_loss = self.criterion(noise_pred, noise)
                car_pixel_loss = self.criterion(
                    noise_pred * interpolate_boxes,
                    noise * interpolate_boxes
                )
                batch_loss = all_image_loss + self.factor * car_pixel_loss
                
                batch_loss.backward()
                self.optimizer.step()
                
                epoch_loss += batch_loss.item()
                img_loss += all_image_loss.item()
                car_loss += car_pixel_loss.item()

            epoch_loss /= (i+1)
            img_loss /= (i+1)
            car_loss /= (i+1)
            metrics = {
                "train_glob_loss": epoch_loss, 
                "train_car_pixel_loss": car_loss, 
                "train_image_loss": img_loss, 
                "step": epoch_idx
            }
            tqdm_bar.set_postfix(train_loss=f"{epoch_loss:.4f}")
            if epoch_idx % self.config["training_configuration"]["test_step"] == 0:
                test_car_loss, test_img_loss, test_loss = self.evaluate(True)
                metrics["test_glob_loss"] = test_loss
                metrics["test_car_pixel_loss"] = test_car_loss
                metrics["test_image_loss"] = test_img_loss
                tqdm_bar.set_postfix(test_loss=f"{test_loss:.4f}", refresh=False)
            self.update_metrics(metrics)
        self.run.finish()
    
    def evaluate(self, show=False):
        """
        Evaluate the fine-tuned stable diffusion model on the test dataset

        Parameters
        ----------
        show : bool, optional
            Flag indicating whether to show or not the generated images during training. The default is False.

        Returns
        -------
        car_loss : float
            The loss observed on the cars's pixels.
        img_loss : float
            The loss on the all image.
        val_loss : float
            The validation loss correspond to img_loss + car_loss * self.factor.

        """
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
        )
        self.unet.eval()
        val_loss = 0.0
        car_loss = 0.0
        img_loss = 0.0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_loader), desc="Evaluation"):
                _, images, boxes, input_ids, attention_mask = batch
                images = images.to(self.device)
                boxes = boxes.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
            
                latents = self.vae.encode(images).latent_dist.sample() \
                    * self.vae.config.scaling_factor
                encoder_hidden_states = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    self.scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device
                ).long()
                noisy_latents = self.scheduler.add_noise(
                    latents, 
                    noise, 
                    timesteps
                )
                
                noise_pred = self.unet(
                    noisy_latents, 
                    timesteps,
                    encoder_hidden_states
                ).sample
                
                interpolate_boxes = F.interpolate(
                    boxes.to(latents.device),
                    size=latents.shape[-2:], 
                    mode="nearest"
                )
                
                all_image_loss = self.criterion(noise_pred, noise)
                car_pixel_loss = self.criterion(
                    noise_pred * interpolate_boxes, 
                    noise * interpolate_boxes
                )
                
                batch_loss = all_image_loss + self.factor * car_pixel_loss
                
                val_loss += batch_loss.item()
                car_loss += car_pixel_loss.item()
                img_loss += all_image_loss.item()
                
            val_loss /= (i+1)
            car_loss /= (i+1)
            img_loss /= (i+1)
            
            if show:
                self.test_show()    
                
            return car_loss, img_loss, val_loss            
    
    def update_metrics(self, metrics):
        step_key = "step"
        test_key = "test_glob_loss"
        for key in metrics.keys():
            if key != step_key:
                self.run.log({key:metrics[key]}, step=metrics[step_key])
        if test_key in metrics: 
            test_loss = metrics[test_key]
            if self.best_loss > test_loss:
                self.best_loss = test_loss
                self.best_iter = metrics[step_key]
                self.save_model()
            
    def inference(self, prompt):
        if not self.model_loaded:
            self.load_model(False)
        self.unet.eval()
        with torch.no_grad():
            start = time()
            image = self.pipe(prompt).images[0] 
            end = time()
        return image, (end - start)

if __name__ == "__main__":
    config_file = "../Config/sd_finetuning.yaml"
    trainer = Trainer(config_file)
    trainer.train()
        