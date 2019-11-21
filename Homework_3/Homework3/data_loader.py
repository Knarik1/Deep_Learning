import os
import glob
import gzip
import numpy as np
from skimage.io import imread
from utils import download, save_in_folders, one_hot_encoded

# file names for the data-set.
filename_x_train = "train-images-idx3-ubyte.gz"
filename_y_train = "train-labels-idx1-ubyte.gz"
filename_x_test = "t10k-images-idx3-ubyte.gz"
filename_y_test = "t10k-labels-idx1-ubyte.gz"

class DataLoader:
    def __init__(self, train_images_dir, val_images_dir, test_images_dir, train_batch_size, val_batch_size,
                 test_batch_size, height_of_image, width_of_image, num_channels, num_classes, flatten):

        # train/val/test data sizes
        self.m_train = 55000
        self.m_val = 5000
        self.m_test = 10000

        self.flatten = flatten

        # setting height/ width/ number of channels/ number of classes
        self.height_of_image = height_of_image
        self.width_of_image = width_of_image
        self.num_channels = num_channels
        self.num_classes = num_classes

        # getting images paths
        self.train_paths = glob.glob(os.path.join(train_images_dir, "**/*.png"), recursive=True)
        self.val_paths = glob.glob(os.path.join(val_images_dir, "**/*.png"), recursive=True)
        self.test_paths = glob.glob(os.path.join(test_images_dir, "**/*.png"), recursive=True)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        # download and load images.
        x_train = self.load_images(filename_x_train, train_images_dir)
        y_train_cls = self.load_cls(filename_y_train, train_images_dir)
        self.x_test = self.load_images(filename_x_test, test_images_dir)
        self.y_test_cls = self.load_cls(filename_y_test, test_images_dir)

        # split into train/validation sets
        self.x_train = x_train[0:self.m_train]
        self.x_val = x_train[self.m_train:]
        self.y_train_cls = y_train_cls[0:self.m_train]
        self.y_val_cls = y_train_cls[self.m_train:]

        # one-hot-encode labels
        self.y_train = one_hot_encoded(self.y_train_cls, self.num_classes)
        self.y_val = one_hot_encoded(self.y_val_cls, self.num_classes)
        self.y_test = one_hot_encoded(self.y_test_cls, self.num_classes)

        # save in class folders
        save_in_folders(path=train_images_dir, images=self.x_train, labels=self.y_train_cls)
        save_in_folders(path=val_images_dir, images=self.x_val, labels=self.y_val_cls)
        save_in_folders(path=test_images_dir, images=self.x_test, labels=self.y_test_cls)

        print("----Data Loader init----")

    def load_images(self, filename, data_dir):
        data = self.load_data(filename, data_dir, 16)

        # getting data with shape (number_of_images, 28, 28)
        images = data.reshape(-1, self.height_of_image, self.width_of_image, self.num_channels)

        return images

    def load_cls(self, filename, data_dir):
        return self.load_data(filename, data_dir, 8)

    def load_data(self, filename, data_dir, offset):
        # download data from the internet
        download(filename=filename, download_dir=data_dir)

        # unzip and read data
        path = os.path.join(data_dir, filename)
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=offset)

        return data

    def load_image(self, path):
        image = imread(path)
        cls = int(path.split('/')[-2])
        label = one_hot_encoded(cls, self.num_classes)

        return image, label

    def batch_data_loader(self, batch_size, file_paths, index, perm=None):
        x_batch = []
        y_cls_batch = []
        file_paths = np.array(file_paths)

        # shuffle file_paths array
        if perm is not None:
            file_paths = file_paths[perm]

        # get mini-batch of paths
        file_paths_batch = file_paths[index: index + batch_size]

        # reading images from paths
        for path in file_paths_batch:
            # getting img array
            im = imread(path)

            #reshape image
            if self.flatten:
                im = im.reshape(self.height_of_image * self.width_of_image * self.num_channels)
            else:
                im = im.reshape(self.height_of_image, self.width_of_image, self.num_channels)

            x_batch.append(im)

            # getting class by splitting path
            cls = int(path.split('/')[-2])
            y_cls_batch.append(cls)

        # converting to np.array and normalizing
        x_batch = np.array(x_batch)/255.
        y_cls_batch = np.array(y_cls_batch)

        # getting one-hot-encoded labels
        y_batch = one_hot_encoded(y_cls_batch, self.num_classes)

        return x_batch, y_batch, y_cls_batch

    def train_data_loader(self, index, perm=None):
        return self.batch_data_loader(self.train_batch_size, self.train_paths, index, perm=perm)

    def val_data_loader(self, index, perm=None):
        return self.batch_data_loader(self.val_batch_size, self.val_paths, index, perm=perm)

    def test_data_loader(self, index, perm=None):
        return self.batch_data_loader(self.test_batch_size, self.test_paths, index)