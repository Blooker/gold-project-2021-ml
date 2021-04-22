import os

import pandas as pd
import torch

from torch.utils.data import Dataset

from skimage import io


class UnityDataset(Dataset):
    def __init__(self, lit_folder, unlit_folder, depth_folder, csv_file, root_dir, img_size, patch_size, light_data_size, transform=None):
        self.root_dir = root_dir
        self.lit_folder = lit_folder
        self.unlit_folder = unlit_folder
        self.depth_folder = depth_folder

        self.annotations = pd.read_csv(self.root_dir + "/" + csv_file, header=None).values

        self.transform = transform

        self.img_size = img_size
        self.patch_size = patch_size
        self.light_data_size = light_data_size

        self.patch_dim = (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1])

        self.num_images = 27516

    def __len__(self):
        return len(self.annotations) * self.patch_dim[0] * self.patch_dim[1]

    def __getitem__(self, index):

        patch = self.patch_from_index(index)
        # patch[0] = image index
        # patch[1] = col that we'll take the patch from
        # patch[2] = row that we'll take the patch from
        # patch[3] = index of the patch from top left to bottom right

        filename = "{}.png".format(patch[0])

        x_reg_filename = self.unlit_folder + "/" + filename
        x_dep_filename = self.depth_folder + "/" + filename
        y_filename = self.lit_folder + "/" + filename

        x_reg_path = os.path.join(self.root_dir, x_reg_filename)
        x_dep_path = os.path.join(self.root_dir, x_dep_filename)
        y_img_path = os.path.join(self.root_dir, y_filename)

        x_reg_image = io.imread(x_reg_path)
        x_dep_image = io.imread(x_dep_path)
        y_image = io.imread(y_img_path)

        patch_x_bounds = patch[1]*self.patch_size[0], (patch[1]+1)*self.patch_size[0]
        patch_y_bounds = patch[2]*self.patch_size[1], (patch[2]+1)*self.patch_size[1]

        # Get rid of alpha channel and crop to desired patch size
        x_reg_image = x_reg_image[patch_y_bounds[0]:patch_y_bounds[1], patch_x_bounds[0]:patch_x_bounds[1], :3]
        y_image = y_image[patch_y_bounds[0]:patch_y_bounds[1], patch_x_bounds[0]:patch_x_bounds[1], :3]

        # Get rid of all but one channel (greyscale) and crop to desired patch size
        x_dep_image = x_dep_image[patch_y_bounds[0]:patch_y_bounds[1], patch_x_bounds[0]:patch_x_bounds[1], :1]

        if self.transform:
            x_reg_image = self.transform(x_reg_image)
            x_dep_image = self.transform(x_dep_image)
            y_image = self.transform(y_image)

        # Get patch index
        x_patch = patch[3] / (self.patch_dim[0] * self.patch_dim[1])

        # Get corresponding pos and light data
        x_pos = torch.Tensor(self.annotations[patch[0], :-self.light_data_size])
        x_light = torch.Tensor(self.annotations[patch[0], -self.light_data_size:])

        return x_reg_image, x_dep_image, x_patch, x_pos, x_light, y_image

    def patch_from_index(self, index):
        patches = self.patch_dim[0] * self.patch_dim[1]

        img_index = index // patches

        col_index = index % self.patch_dim[0]
        row_index = (index - (patches*img_index)) // self.patch_dim[0]

        patch_index = (row_index * self.patch_dim[0]) + col_index

        return img_index, col_index, row_index, patch_index
