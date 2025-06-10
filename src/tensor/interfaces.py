import pickle
from abc import ABC, abstractmethod

from PIL.Image import Image

from src.config.config import Config


class ITensorHandler(ABC):
    def __init__(self, config: Config):
        self._config = config

    @abstractmethod
    def zeros(self, shape):
        raise NotImplementedError()

    @abstractmethod
    def image_to_array(self, image: Image):
        raise NotImplementedError()

    @abstractmethod
    def scalar_to_array(self, a):
        raise NotImplementedError()

    @abstractmethod
    def resize_array_1d(self, data, n):
        raise NotImplementedError()

    @abstractmethod
    def resize(self, value, shape):
        raise NotImplementedError()

    @abstractmethod
    def normalise(self, x):
        raise NotImplementedError()

    @abstractmethod
    def concatenate(self, a, b, axis):
        raise NotImplementedError()

    @abstractmethod
    def multiply(self, a, b):
        raise NotImplementedError()

    @abstractmethod
    def exp(self, z):
        raise NotImplementedError()

    @abstractmethod
    def log(self, z):
        raise NotImplementedError()

    @abstractmethod
    def sum(self, data):
        raise NotImplementedError()

    @abstractmethod
    def is_nan(self, value):
        raise NotImplementedError()

    @abstractmethod
    def fill(self, array, value):
        raise NotImplementedError()

    @abstractmethod
    def relu(self, value):
        raise NotImplementedError()

    @abstractmethod
    def diag(self, value):
        raise NotImplementedError()

    @abstractmethod
    def d_relu(self, value):
        raise NotImplementedError()

    @abstractmethod
    def randn(self, shape):
        raise NotImplementedError()

    @abstractmethod
    def sqrt(self, value):
        raise NotImplementedError()

    @abstractmethod
    def as_tensor(self, value):
        raise NotImplementedError()

    @abstractmethod
    def clone(self, value):
        raise NotImplementedError()