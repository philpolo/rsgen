#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:27:06 2026

@author: polo
"""

import os
import cv2
import sys 
import re
import argparse
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import tensorflow as tf

sys.path.append("..")
sys.path.append("../../")
sys.path.append("..")
sys.path.append("../Communication")

from gbsed.pipeline.pipeline import GBSED
from text2image.sd_trainer import Trainer
from utils.datasetGenerator import sg2text

import roadscene2vec
from roadscene2vec.util.config_parser import configuration

sys.modules['util'] = roadscene2vec.util

class phy_layer_enhancer(GBSED):
    
    def __init__(
            self, 
            gbsed_config:configuration, 
            sd_config_filename:os.PathLike,
            com_model_endpoint, 
            batch_size:int=16
        ):
        super(phy_layer_enhancer, self).__init__(gbsed_config, com_model_endpoint, batch_size)
        self.sd_config_filename = sd_config_filename
        self.rs_generetor = Trainer(self.sd_config_filename)
        self.rs_generetor.load_model(training=False)  
        self.text_generator = sg2text(gbsed_config)
    
    def _image_storage_format_(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        shape = (img.shape[0] // 8, img.shape[1] // 8)
        img = cv2.resize(img, shape, cv2.INTER_LANCZOS4)
        return np.array(img.shape + tuple(img.flatten()), dtype=np.float16)
    
    def _image_format_loading_(self, array):
        shape, img = tuple(array[:2].astype(int)), array[2:]
        img = img.reshape(shape)
        img = img.astype(np.uint8)
        new_shape = (shape[0] * 8, shape[1] * 8)
        img = cv2.resize(img, new_shape, cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    
    def _process_image_(self, img):
        array = self._image_storage_format_(img)
        input_bits = self._to_bits_array_(array)
        return {
            "shape":tuple(array[:2].astype(int)),
            "flattened_image":array[2:],
            "input_bits":input_bits,
            "prepared_bits":self._prepare_bits_for_model_(input_bits)
        }
    
    def _image_reconstruction_(self, received_bits, initial_processed_img):
        received_array = self._to_float_array_(received_bits)
        received_shape = tuple(received_array[:2].astype(int))
        received_flattened_image = received_array[2:]
        if initial_processed_img["shape"] == received_shape \
            and np.allclose(initial_processed_img['flattened_image'],received_flattened_image):
            return self._image_format_loading_(received_array)
        else:
            print("Truncated file\n", file=sys.stderr)
            
    def image_transmission(self, img):
        initial_processed_img = self._process_image_(img)
        start = time()
        b, b_hat = self.model(
            self.batch_size, 
            self.ebno, 
            initial_processed_img['prepared_bits']
        )
        end = time()
        received_bits = self._recover_original_bits_(
            b_hat, 
            initial_processed_img['input_bits'].size
        ).cpu().numpy()
        received_bits = np.asarray(received_bits, dtype=np.uint8)
        return self._image_reconstruction_(received_bits, initial_processed_img), (end - start)
    
    def e2e(self):
        path = self.config.location_data['input_path']
        if self.config.loading_type == "pickle":
            location = self.data.dataset_path
            if not os.path.exists(location):
                os.mkdir(location)
        elif self.config.loading_type == "folder":
            location = path
            
        scene_graphs = self.data.scene_graphs
        folder_names = self.data.folder_names
        pattern = r'(\d+)'
        prog = re.compile(pattern)
        folder_paths = {
            int(prog.match(f).group(0)):os.path.join(location, f) for f in folder_names
        }
        keys = list(scene_graphs.keys())
        sg_durations, img_durations, gen_durations = [], [], []
        
        p_bar = tqdm(range(len(keys)))
        for i in p_bar:
            key = keys[i]
            p_bar.set_postfix_str(f"folder={key}")
            folder_name = folder_names[i]
            seq_folder = os.path.join(location, folder_name)
            if not os.path.exists(seq_folder):
                os.mkdir(seq_folder)
            to_store_folder = os.path.join(seq_folder, "encoded_files")
            if not os.path.exists(to_store_folder):
                os.mkdir(to_store_folder)
            received_folder = os.path.join(seq_folder, "received_files")
            if not os.path.exists(received_folder):
                os.mkdir(received_folder)
            received_images_folder = os.path.join(folder_paths[key], "received_images")
            if not os.path.exists(received_images_folder):
                os.mkdir(received_images_folder)
            gen_images_folder = os.path.join(folder_paths[key], "generated_images")
            if not os.path.exists(gen_images_folder):
                os.mkdir(gen_images_folder)
            images_folder = os.path.join(folder_paths[key], "raw_images")
            filenames = os.listdir(images_folder)
            
            sequence = scene_graphs[key]
            for seq_file_num in sequence.keys():
                sg = sequence[seq_file_num]
                rec_sg, sg_time = self.sg_transmission(sg)
                sg.visualize(os.path.join(to_store_folder, str(seq_file_num) + ".png"))
                if not rec_sg is None:
                    rec_sg.visualize(os.path.join(received_folder, str(seq_file_num) + ".png"))
                sg_durations.append(sg_time)
                
                filename = [i for i in filenames if i.endswith(str(seq_file_num) + ".jpg")][0]
                filenames.remove(filename)
                
                caption = self.text_generator.scene_graph_to_prompt(sg)
                image, gen_time = self.rs_generetor.inference(caption)
                image.save(os.path.join(gen_images_folder, filename))
                gen_durations.append(gen_time)
                
                img = cv2.cvtColor(
                    cv2.imread(os.path.join(images_folder, filename), cv2.IMREAD_COLOR), 
                    cv2.COLOR_BGR2RGB
                )
                try:
                    rec_image, img_time = self.image_transmission(img)
                    rec_image = Image.fromarray(rec_image)
                    rec_image.save(os.path.join(received_images_folder, filename))
                    img_durations.append(img_time)
                    
                except AttributeError:
                    sg_durations.remove(sg_time)
                    gen_durations.remove(gen_time)
                    continue
                
        return pd.DataFrame(
            {
                "sg_duration":sg_durations, 
                "img_duration":img_durations, 
                "gen_duration":gen_durations
            }
        )
    
    
def main(learning_filename, pipe, ebno, time_file):
    pipe._set_ebno(ebno)   
    df = pipe.e2e()
    df.to_csv(time_file, index=False)
    

def main_parser():
    parser = argparse.ArgumentParser(
        description="The complete pipeline from scene graph extraction"\
            "semantic enconding, wireless communication, semantic decoding, "\
            "to collision prediction."
    )
        
    parser.add_argument(
        "--extraction_filename", 
        type=str, 
        default="../Config/pipeline_extraction.yaml", 
        help="Path to scene graph extraction configuration file."
    )
                                     
    parser.add_argument(
        "--learning_filename", 
        type=str,
        default="../Config/pipeline_learning.yaml", 
        help="Path to the model learning configuration file."
    )
    
    parser.add_argument(
        "--sd_config_filename", 
        type=str,
        default="../Config/sd_finetuning.yaml", 
        help="Path to the image generator model configuration file."
    )
    
    parser.add_argument(
        "--com_model_endpoint", 
        type=str, 
        default="../../gbsed/Communication/weights/Neural_Demaper.h5", 
        help="Path to the communication model checkpoint."
    )
    
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="../../Data/Outputs/outputs.csv", 
        help="Path to save the outputs"
    )
    
    parser.add_argument(
        "--time_file", 
        type=str, 
        default="../../Data/Outputs/transfert_times.csv", 
        help="Path to save the transfert times dataframe"
    )
    
    return parser.parse_args()
        
            
if __name__ == "__main__":
    from PIL import Image
    args = main_parser()
    extraction_config = configuration(
        args.extraction_filename, 
        from_function=True
    )
    pipe = phy_layer_enhancer(
        extraction_config, 
        args.sd_config_filename,
        args.com_model_endpoint, 
        batch_size=32
    )
    ebno = tf.constant(25, tf.float32)
    main(args.learning_filename, pipe, ebno, args.time_file)
