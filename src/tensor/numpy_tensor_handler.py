import numpy as np
import numpy.random
from PIL.Image import Image

from src.tensor.interfaces import ITensorHandler


class _NumpyTensorHandler(ITensorHandler):

    def zeros(self, shape):
        return np.zeros(shape)

    def image_to_array(self, image: Image):
        return np.array(image)

    def scalar_to_array(self, a):
        return np.array([a])

    def resize_array_1d(self, x, n):
        x.resize((1, n))

    def resize(self, value, shape):
        value.resize(shape)

    def normalise(self, x):
        return x / 255

    def concatenate(self, a, b, axis):
        return np.concatenate((a, b), axis=axis)

    def multiply(self, a, b):
        return np.dot(a, b)

    def exp(self, x):
        return np.exp(x)

    def log(self, x):
        return np.log(x)

    def sum(self, x):
        return np.sum(x)

    def is_nan(self, value):
        return np.isnan(value)

    def fill(self, array, value):
        array.fill(value)

    def relu(self, value):
        return np.maximum(0, value)

    def d_relu(self, value):
        return (value > 0).astype(float)

    def randn(self, shape):
        return numpy.random.rand(*shape)

    def sqrt(self, value):
        return numpy.sqrt(value)

    def as_tensor(self, value):
        return value

    def diag(self, value):
        return np.diag(value)

    def clone(self, value):
        return value.copy()
