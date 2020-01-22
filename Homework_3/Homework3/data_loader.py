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
                 test_batch_size, height_of_image, width_of_image, num_channels, num_classes, model):

        self.config = {
            "m_train": 55000,
            "m_val": 5000,
            "m_test": 10000,
            "train_batch_size": train_batch_size,
            "val_batch_size": val_batch_size,
            "test_batch_size": test_batch_size,
            "img_h": height_of_image,
            "img_w": width_of_image,
            "img_chls": num_channels,
            "num_cls": num_classes,
            "model": model
        }

        # getting images paths
        self.train_paths = glob.glob(os.path.join(train_images_dir, "**/*.png"), recursive=True)
        self.val_paths = glob.glob(os.path.join(val_images_dir, "**/*.png"), recursive=True)
        self.test_paths = glob.glob(os.path.join(test_images_dir, "**/*.png"), recursive=True)

        # download and load images.
        x_train = self.load_images(filename_x_train, train_images_dir)
        y_train_cls = self.load_cls(filename_y_train, train_images_dir)
        self.x_test = self.load_images(filename_x_test, test_images_dir)
        self.y_test_cls = self.load_cls(filename_y_test, test_images_dir)

        # split into train/validation sets
        self.x_train = x_train[0:self.config["m_train"]]
        self.x_val = x_train[self.config["m_train"]:]
        self.y_train_cls = y_train_cls[0:self.config["m_train"]]
        self.y_val_cls = y_train_cls[self.config["m_train"]:]

        # one-hot-encode labels
        self.y_train = one_hot_encoded(self.y_train_cls, self.config["num_cls"])
        self.y_val = one_hot_encoded(self.y_val_cls, self.config["num_cls"])
        self.y_test = one_hot_encoded(self.y_test_cls, self.config["num_cls"])

        # save in class folders
        save_in_folders(path=train_images_dir, images=self.x_train, labels=self.y_train_cls)
        save_in_folders(path=val_images_dir, images=self.x_val, labels=self.y_val_cls)
        save_in_folders(path=test_images_dir, images=self.x_test, labels=self.y_test_cls)

        print("----Data Loader init----")

    def load_images(self, filename, data_dir):
        data = self.load_data(filename, data_dir, 16)

        # getting data with shape (number_of_images, 28, 28)
        images = data.reshape(-1, self.config["img_h"], self.config["img_w"], self.config["img_chls"])

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
        label = one_hot_encoded(cls, self.config["num_cls"])

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
            if self.config["model"] == 'DNN':
                im = im.reshape(self.config["img_h"] * self.config["img_w"] * self.config["img_chls"])
            elif self.config["model"] == 'RNN':
                im = im.reshape(self.config["img_h"], self.config["img_w"])
            else:
                im = im.reshape(self.config["img_h"], self.config["img_w"], self.config["img_chls"])

            x_batch.append(im)

            # getting class by splitting path
            cls = int(path.split('/')[-2])
            y_cls_batch.append(cls)

        # converting to np.array and normalizing
        x_batch = np.array(x_batch)/255.
        y_cls_batch = np.array(y_cls_batch)

        # getting one-hot-encoded labels
        y_batch = one_hot_encoded(y_cls_batch, self.config["num_cls"])

        return x_batch, y_batch, y_cls_batch

    def train_data_loader(self, index, perm=None):
        return self.batch_data_loader(self.config["train_batch_size"], self.train_paths, index, perm=perm)

    def val_data_loader(self, index, perm=None):
        return self.batch_data_loader(self.config["val_batch_size"], self.val_paths, index, perm=perm)

    def test_data_loader(self, index, perm=None):
        return self.batch_data_loader(self.config["test_batch_size"], self.test_paths, index, perm=perm)