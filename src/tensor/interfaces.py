from abc import ABC, abstractmethod

from PIL.Image import Image

from src.tensor.tensor_handler_config import TensorHandlerConfig


class ITensorHandler(ABC):
    def __init__(self, log, config: TensorHandlerConfig):
        self._config = config
        self._log = log

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
    def normalise(self, x):
        raise NotImplementedError()

    @abstractmethod
    def concatenate(self, a, b, axis):
        raise NotImplementedError()

    @abstractmethod
    def save(self, data, path):
        raise NotImplementedError()

    @abstractmethod
    def load(self, path):
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
    def unpack_checkpoint(self, checkpoint_data):
        raise NotImplementedError()

    @abstractmethod
    def is_nan(self, value):
        raise NotImplementedError()

    @abstractmethod
    def fill(self, array, value):
        raise NotImplementedError()
