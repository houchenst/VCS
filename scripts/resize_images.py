from torch.utils.data import Dataset
import torch
import numpy as np
import PIL
from PIL import Image
import os
from tqdm import tqdm

'''
This script is used to resize the training data for DeepFashion
The README for the original dataset said that the images were resized so that their larger
dimension was 300. I found that that wasn't the case, so this script will actually perform
that resizing. It also adds black padding so that all images are 300x300.

To run this script, set ROOT_DIR to be the location of your deep fashion datset. The img/ 
folder should be renamed to original_img/ and a new empty img/ folder should be created. The
empty folder will be filled by this script.
'''

ROOT_DIR = "F:\\DeepFashionDataset"

if __name__ == "__main__":
    with open(os.path.join(ROOT_DIR, "list_attr_img.txt"), 'r') as img_file:
        img_lines = img_file.readlines()[2:]
        print("Resizing DeepFashion images...")
        for line in tqdm(img_lines):
            splt_line = line.split()
            filename = os.path.join(ROOT_DIR, splt_line[0])

            # make subdirectory if it doesn't exist
            if not os.path.exists(os.path.dirname(filename)):
                os.mkdir(os.path.dirname(filename))

            # replace /img with /original_img
            old_filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(filename))), "original_img", os.path.basename(os.path.dirname(filename)), os.path.basename(filename))
            
            image = Image.open(old_filename)
            height = image.size[0]
            width = image.size[1]
            max_dim = max(height,width)
            new_height = height * 300 // max_dim
            new_width = width * 300 // max_dim

            resized_image = image.resize((new_height, new_width), resample=PIL.Image.BILINEAR)

            # create a new 300x300 black RGB image
            final_image = Image.new("RGB", (300,300))

            y_offset = (300 - resized_image.size[0]) // 2
            x_offset = (300 - resized_image.size[1]) // 2

            # paste the resized image on to the black background
            final_image.paste(resized_image, (y_offset, x_offset))
            final_image.save(filename)
    print("Done.")
            