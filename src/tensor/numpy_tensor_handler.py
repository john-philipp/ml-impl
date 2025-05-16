import numpy as np
from PIL.Image import Image

from src.tensor.interfaces import _ITensorHandler


class _NumpyTensorHandler(_ITensorHandler):
    def zeros(self, shape):
        return np.zeros(shape)

    def image_to_array(self, image: Image):
        return np.array(image)

    def scalar_to_array(self, a):
        return np.array([a])

    def resize_array_1d(self, x, n):
        x.resize((1, n))

    def normalise(self, x):
        return x / 255

    def concatenate(self, a, b, axis):
        return np.concatenate((a, b), axis=axis)

    def save(self, x, path):
        np.save(path, x)

    def load(self, path):
        return np.load(path)

    def multiply(self, a, b):
        return np.dot(a, b)

    def exp(self, x):
        return np.exp(x)

    def log(self, x):
        return np.log(x)

    def sum(self, x):
        return np.sum(x)

    def unpack_checkpoint(self, checkpoint_data):
        return checkpoint_data[:-1], checkpoint_data[-1]
