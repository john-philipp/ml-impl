import logging
import time

import numpy as np
import torch
import os

from PIL import ImageOps, Image
from logging import basicConfig, getLogger

from torchvision import transforms

basicConfig(level=logging.INFO)
log = getLogger(__name__)


DEFAULT_IMAGE_WIDTH = 224
DEFAULT_IMAGE_HEIGHT = 224


class Context:

    CUDA = False

    def __init__(self, dimensions, alpha, image_width, image_height):
        self.alpha = alpha  # Learning rate.
        self.dimensions = dimensions
        self.datas = []

        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = None

        # Weights and bias.
        if Context.CUDA:
            self.w = torch.zeros(self.dimensions, device=device)
        else:
            self.w = np.zeros(self.dimensions)

        self.b = 0

        self.m = 0  # Sample count.
        self.a = None  # Predictions. Activation function (sigmoid): 1 / (1 + e^-z)
        self.x = None  # Inputs.
        self.y = None  # Truth.

        self.j = None  # Average cost over all samples.
        self.z = None  # Logit (raw score prior to activation: z = w.T dot x + b

        # Derivatives of J (cost) wrt to variable.
        self.dl_dz = None
        self.dw = None
        self.db = None

    @staticmethod
    def load_image(image_path):
        return Image.open(image_path)

    @staticmethod
    def resize_image(image, width, height, padding_rgb=(0, 0, 0)):
        image.thumbnail((width, height))
        delta_w = width - image.width
        delta_h = height - image.height
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        return ImageOps.expand(image, padding, fill=padding_rgb)

    @staticmethod
    def to_array(image):
        if Context.CUDA:
            tensor = transforms.ToTensor()(image)
            tensor.to(device=device)
            return tensor
        else:
            return np.array(image)

    def set_image_size(self, width, height):
        self.image_width = width
        self.image_height = height

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def load_image_data(self, file_path):
        image = self.load_image(file_path)
        image = self.resize_image(image, self.image_width, self.image_height)
        image_data = self.to_array(image)
        if self.CUDA:
            image_data.resize_(self.dimensions)
        else:
            image_data.resize((1, self.dimensions))
        return image_data

    def load_data(self, path, label):
        files = os.listdir(path)
        files.sort()

        dm = len(files)
        if batch_size and batch_size < dm:
            files = files[:batch_size]
            dm = batch_size
        self.m += dm

        # Note, without batching this will eventually incur out of memory errors.
        if self.CUDA:
            x = torch.zeros((self.dimensions, dm), device=device)
        else:
            x = np.zeros((self.dimensions, dm), dtype=np.float32)

        i = 0
        for file in files:
            file_path = os.path.join(path, file)
            image_data = self.load_image_data(file_path)
            x[:, i] = image_data
            i += 1

        self.datas.append((label, x))

    def accumulate_data(self):
        assert self.datas, "Must load data prior."
        assert self.m == sum([x.shape[1] for _, x in self.datas]), "Failed consistency check!"

        # Note, without batching this will eventually incur out of memory errors.
        if self.CUDA:
            self.x = torch.zeros((self.dimensions, self.m), device=device)
            self.y = torch.zeros(self.m, device=device)
        else:
            self.x = np.zeros((self.dimensions, self.m), dtype=np.float32)
            self.y = np.zeros(self.m, dtype=np.int8)

        offset = 0
        for label, x in self.datas:
            m = x.shape[1]
            for j in range(0, m):
                self.x[:, offset + j] = x[:, j]
                self.y[offset + j] = label
            offset += m

    def normalise_data(self):
        if self.CUDA:
            # torchvision does this on its own.
            return
        self.x /= 255

    def save_checkpoint(self, path):
        log.info("Trying to save checkpoint...")
        if self.CUDA:
            log.info(f"Saving checkpoint (CUDA): {path}")
            torch.save(torch.cat((ctx.w, torch.tensor([ctx.b], device=device)), dim=0), path)
        else:
            log.info(f"Saving checkpoint (numpy): {path}")
            np.save(path, np.concatenate((self.w, np.array([self.b])), axis=0))
        log.info("Done saving checkpoint.")

    def try_load_checkpoint(self, path):
        log.info("Trying to load checkpoint...")
        if not os.path.isfile(path):
            log.warning(f"No such checkpoint: {path}")
            return
        if self.CUDA:
            log.info(f"Loading checkpoint (CUDA): {path}")
            tensor = torch.load(path)
            assert tensor.shape[0] == self.dimensions + 1
            self.w = tensor[:-1]
            self.b = tensor[-1].item()
        else:
            log.info(f"Loading checkpoint (numpy): {path}")
            array = np.load(path)
            assert array.shape[0] == self.dimensions + 1
            self.w = array[:-1]
            self.b = array[-1]
        log.info("Done loading checkpoint.")

    def update_z(self):
        if self.CUDA:
            self.z = torch.matmul(self.w, self.x)
        else:
            self.z = np.dot(self.w.T, self.x) + self.b

    def update_a(self):
        if self.CUDA:
            self.a = 1 / (1 + torch.exp(-self.z))
        else:
            self.a = 1 / (1 + np.exp(-self.z))

    def update_j(self):
        if self.CUDA:
            self.j = (-1 / self.m) * (
                torch.matmul(self.y, torch.log(self.a)) + torch.matmul(1 - self.y, torch.log(1 - self.a)))
        else:
            self.j = (-1 / self.m) * (np.dot(self.y, np.log(self.a).T) + np.dot(1 - self.y.T, np.log(1 - self.a).T))

    def update_dl_dz(self):
        self.dl_dz = self.a - self.y

    def update_dj_dw(self):
        if self.CUDA:
            self.dw = (1 / self.m) * torch.matmul(self.x, self.dl_dz)
        else:
            self.dw = (1 / self.m) * np.dot(self.x, self.dl_dz.T)

    def update_dj_db(self):
        if self.CUDA:
            self.db = (1 / self.m) * torch.sum(self.dl_dz)
        else:
            self.db = (1 / self.m) * np.sum(self.dl_dz)

    def update_w(self):
        self.w -= self.alpha * self.dw

    def update_b(self):
        self.b -= self.alpha * self.db

    def train_epoch(self):
        # Forward propagation.
        self.update_z()
        self.update_a()
        self.update_j()

        # Backward propagation.
        self.update_dl_dz()
        self.update_dj_db()
        self.update_dj_dw()

        # Update weights and bias.
        self.update_w()
        self.update_b()

        return self.j

    def infer(self, image_path):
        self.m = 1
        if self.CUDA:
            self.x = torch.zeros((self.dimensions, self.m), device=device)
        else:
            self.x = np.zeros((self.dimensions, self.m), dtype=np.float32)
        image_data = self.load_image_data(image_path)

        # self.x[:, 0] = image_data
        self.x[:, 0] = torch.rand(self.dimensions)
        self.normalise_data()

        self.update_z()
        self.update_a()

        return self.a


if __name__ == '__main__':
    log.info("Starting...")

    cuda = True
    # cuda = False
    # train = True  # Else infer.
    train = False
    learning_rate = 0.001  # == alpha.
    points_sqrt = 224
    batch_size = 100
    epochs = 21000
    log_every = 1000

    if cuda:
        Context.CUDA = cuda
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            log.error("CUDA unavailable.")
            raise ValueError()

    image_width = points_sqrt  # Resize images to this width.
    image_height = points_sqrt  # Resize images to this height.
    pixels_rgb = image_width * image_height  # Coloured pixels in image.
    pixels_raw = pixels_rgb * 3  # Individual rgb ints across all pixels.
    checkpoint_path = "checkpoint" + (".pt" if cuda else ".npy")

    ctx = Context(pixels_raw, learning_rate, image_width, image_height)
    if train:
        ctx.load_data("../.datasets/cats_dogs/dogs", 0)
        ctx.load_data("../.datasets/cats_dogs/cats", 1)
        ctx.accumulate_data()
        ctx.normalise_data()

    ctx.try_load_checkpoint(checkpoint_path)

    t0_total = time.time()
    t0 = t0_total
    try:
        if train:
            log.info("Training...")
            for epoch in range(epochs):
                epoch += 1
                cost = ctx.train_epoch()
                t1 = time.time()
                if epoch % log_every == 0:
                    log.info(f"Epoch {epoch} done after {t1 - t0:.3f}s: cost={cost:.4f}")
                    t0 = t1
        else:
            # Infer.
            log.info("Inferring...")
            for type_ in ["dog", "cat"]:
                dir_path = f"../.datasets/cats_dogs/{type_}s/"
                files = os.listdir(dir_path)
                files.sort()
                for file in files[batch_size:batch_size + 10]:
                    print(f"{type_}: {ctx.infer(os.path.join(dir_path, file))}")
    except KeyboardInterrupt:
        pass

    ctx.save_checkpoint(checkpoint_path)

    log.info(f"Done after {time.time() - t0_total:.3f}s.")
