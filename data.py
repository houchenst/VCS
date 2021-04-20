from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepFashionDataset(Dataset):
    '''
    Loads the DeepFashion Dataset
    Returns image path, attribute pairs
    Loading the images themselves would take too much memory
    '''

    def __init__(self, root_dir, set_type='train', max_attrs=21):
        '''
        root_dir - the directory containing the dataset. Should contain the following:
                   eval/list_eval_partition.txt   (determines train/test/val split)
                   img/                           (has subdirectories with the images)
                   list_attr_cloth.txt            (lists clothing attributes in order)
                   list_attr_img.txt              (lists the image paths and their attributes)
        set_type - The type of dataset this is, 'train', 'test', or 'val'
        max_attrs - the maximum number of attributes to be store for a particular example
        '''
        self.root_dir = root_dir
        self.set_type = set_type
        self.max_attrs = max_attrs

        # LOAD ATTRIBUTE ENCODINGS

        # list of attributes (in order)
        self.attrs = ["none"]
        # dictionary mapping attributes to their indices/encodings
        self.attr2num = {"none":0}
        next_attr_num = 1
        with open(os.path.join(self.root_dir, "list_attr_cloth.txt"), 'r') as attr_file:
            attr_lines = attr_file.readlines()[2:]
            for line in attr_lines:
                splt_line = line.split()
                splt_line = [x for x in splt_line if x != ""]
                splt_line = splt_line[:-1]
                new_attr = " ".join(splt_line)
                self.attrs.append(new_attr)
                self.attr2num[new_attr] = next_attr_num
                next_attr_num += 1

        # LOAD IMAGE PATHS AND THEIR ATTRIBUTES

        self.img_paths = []
        self.img_attrs = []
        self.masks = []

        first = True
        with open(os.path.join(self.root_dir, "eval\\list_eval_partition.txt"), 'r') as partition_file:
            file_set = []
            partition_lines = partition_file.readlines()[2:]
            for line in partition_lines:
                splt_line = line.split()
                if splt_line[-1] == self.set_type:
                    file_set.append(splt_line[0])
            # convert to set for faster queries
            file_set = set(file_set)
            with open(os.path.join(self.root_dir, "list_attr_img.txt"), 'r') as img_file:
                img_lines = img_file.readlines()[2:]
                for line in img_lines:
                    splt_line = line.split()
                    filename = splt_line[0]
                    if filename in file_set:
                        file_attrs = splt_line[1:]
                        file_attrs = [i+1 for i,x in enumerate(file_attrs) if x == "1"]
                        # we only want images that have at least one attribute
                        if len(file_attrs) == 0:
                            continue
                        # Use 0 for no attribute/padding
                        self.img_paths.append(os.path.join(self.root_dir, filename))
                        self.masks.append(torch.tensor([0]*len(file_attrs) + [1]*(self.max_attrs - len(file_attrs)), dtype=bool))
                        file_attrs = torch.tensor(file_attrs + [0]*(self.max_attrs - len(file_attrs)))
                        self.img_attrs.append(file_attrs)
        # print(self.img_paths[0])
        # print(self.img_attrs[100])
        # print(self.masks[100])
        # for x in self.img_attrs[100]:
        #     if x != -1:
        #         print(x)
        #         print(self.attrs[x])
        # print(max([len(x) for x in self.img_attrs]))
        self.length = len(self.img_paths)

    def load_image(self, idx):
        '''
        Lazy loading since the whole dataset won't fit in memory
        '''
        image = Image.open(self.img_paths[idx])
        img_tensor = torch.from_numpy(np.array(image))
        img_tensor = img_tensor.type(torch.float32)
        img_tensor /= 255.
        return img_tensor
            
    def __len__(self):
        '''
        Returns the size of the dataset
        '''
        return self.length

    def __getitem__(self, idx):
        '''
        Returns the pair (image path, image attributes) at index idx
        '''
        return self.load_image(idx), self.img_attrs[idx], self.masks[idx]



if __name__ == "__main__":
    a = DeepFashionDataset("F:\\DeepFashionDataset")