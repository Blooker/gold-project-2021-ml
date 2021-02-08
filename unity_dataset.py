import os

from torch.utils.data import Dataset

from skimage import io


class UnityDataset(Dataset):
    def __init__(self, root_dir, img_size, patch_size, transform=None):
        self.root_dir = root_dir

        self.img_size = img_size
        self.patch_size = patch_size

        self.patch_dim = (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1])

        self.transform = transform

        self.num_images = 27516

    def __len__(self):
        return self.num_images * self.patch_dim[0] * self.patch_dim[1]

    def __getitem__(self, index):

        patch = self.patch_from_index(index)
        # patch[0] = image index
        # patch[1] = col that we'll take the patch from
        # patch[2] = row that we'll take the patch from

        x_reg_filename = "regular/{}.png".format(patch[0] + self.num_images)
        x_dep_filename = "depth/{}.png".format(patch[0])
        y_filename = "regular/{}.png".format(patch[0])

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

        return x_reg_image, x_dep_image, y_image

    def patch_from_index(self, index):
        patches = self.patch_dim[0] * self.patch_dim[1]

        img_index = index // patches
        col_index = index % self.patch_dim[0]
        row_index = (index - (patches*img_index)) // self.patch_dim[0]

        return img_index, col_index, row_index
