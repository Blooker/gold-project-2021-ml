import os

import pandas as pd
import torch

from torch.utils.data import Dataset

from skimage import io


class UnityDataset(Dataset):
    def __init__(self, lit_folder, unlit_folder, depth_folder, csv_file, root_dir, img_size, patch_size, transform=None):
        self.root_dir = root_dir
        self.lit_folder = lit_folder
        self.unlit_folder = unlit_folder
        self.depth_folder = depth_folder

        self.annotations = pd.read_csv(self.root_dir + "/" + csv_file, header=None).values

        self.transform = transform

        self.img_size = img_size
        self.patch_size = patch_size

        self.patch_dim = (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1])

        self.num_patches = self.patch_dim[0] * self.patch_dim[1]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        filename = "{}.png".format(index)

        x_reg_filename = self.unlit_folder + "/" + filename
        x_dep_filename = self.depth_folder + "/" + filename
        y_filename = self.lit_folder + "/" + filename

        x_reg_path = os.path.join(self.root_dir, x_reg_filename)
        x_dep_path = os.path.join(self.root_dir, x_dep_filename)
        y_img_path = os.path.join(self.root_dir, y_filename)

        x_reg_image = io.imread(x_reg_path)
        x_dep_image = io.imread(x_dep_path)
        y_image = io.imread(y_img_path)

        x_reg_patches = torch.zeros(size=(self.num_patches, 3, self.patch_size[1], self.patch_size[0]))
        x_dep_patches = torch.zeros(size=(self.num_patches, 1, self.patch_size[1], self.patch_size[0]))
        x_patch_indices = torch.zeros(size=(self.num_patches, 1))

        y_patches = torch.zeros(size=(self.num_patches, 3, self.patch_size[1], self.patch_size[0]))

        patch_index = 0
        for y in range(self.patch_dim[1]):
            for x in range(self.patch_dim[0]):
                patch_x_bounds = x*self.patch_size[0], (x+1)*self.patch_size[0]
                patch_y_bounds = y*self.patch_size[1], (y+1)*self.patch_size[1]

                # Get rid of alpha channel and crop to desired patch size
                x_reg_patch = x_reg_image[patch_y_bounds[0]:patch_y_bounds[1], patch_x_bounds[0]:patch_x_bounds[1], :3]
                y_patch = y_image[patch_y_bounds[0]:patch_y_bounds[1], patch_x_bounds[0]:patch_x_bounds[1], :3]

                # Get rid of all but one channel (greyscale) and crop to desired patch size
                x_dep_patch = x_dep_image[patch_y_bounds[0]:patch_y_bounds[1], patch_x_bounds[0]:patch_x_bounds[1], :1]

                if self.transform:
                    x_reg_patch = self.transform(x_reg_patch)
                    x_dep_patch = self.transform(x_dep_patch)
                    y_patch = self.transform(y_patch)

                x_reg_patches[patch_index] = x_reg_patch
                x_dep_patches[patch_index] = x_dep_patch
                x_patch_indices[patch_index] = patch_index

                y_patches[patch_index] = y_patch

                patch_index += 1

        # Get corresponding pos and light data
        x_param = torch.Tensor(self.annotations[index])
        x_param = x_param.expand(self.num_patches, len(x_param))

        return x_reg_patches, x_dep_patches, x_param, x_patch_indices, y_patches

    def patch_from_index(self, index):
        patches = self.patch_dim[0] * self.patch_dim[1]

        img_index = index // patches

        col_index = index % self.patch_dim[0]
        row_index = (index - (patches*img_index)) // self.patch_dim[0]

        patch_index = (row_index * self.patch_dim[0]) + col_index

        return img_index, col_index, row_index, patch_index
