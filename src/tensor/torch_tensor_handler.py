import logging

import torch
from PIL.Image import Image
from torchvision import transforms

from src.config.config import Config
from src.tensor.interfaces import ITensorHandler


log = logging.getLogger(__name__)


class _TorchTensorHandler(ITensorHandler):

    def __init__(self, config: Config):
        super().__init__(config)

        # Load CUDA.
        if self._config.use_cuda:
            if not torch.cuda.is_available():
                log.error("CUDA unavailable but configured.")
                raise ValueError()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self._device = device

    def zeros(self, shape):
        return torch.zeros(shape, device=self._device)

    def image_to_array(self, image: Image):
        tensor = transforms.ToTensor()(image)
        tensor.to(device=self._device)
        return tensor

    def scalar_to_array(self, a):
        return torch.tensor([a], device=self._device)

    def resize_array_1d(self, x, n):
        x.resize_(n)

    def normalise(self, x):
        # Torchvision normalises image data automatically.
        return x

    def concatenate(self, a, b, axis):
        return torch.cat((a, b), dim=axis)

    def save(self, x, path):
        torch.save(x, path)

    def load(self, path):
        return torch.load(path)

    def multiply(self, a, b):
        return torch.matmul(a, b)

    def exp(self, x):
        return torch.exp(x)

    def log(self, x):
        return torch.log(x)

    def sum(self, x):
        return torch.sum(x)

    def unpack_checkpoint(self, checkpoint_data):
        w = checkpoint_data[:-1]
        w.to(device=self._device)
        b = checkpoint_data[-1].item()
        return w, b

    def is_nan(self, value):
        return torch.isnan(value)

    def fill(self, array, value):
        array.fill_(value)

    def relu(self, value):
        return torch.maximum(value, torch.tensor(0.0))

    def d_relu(self, value):
        return (value > 0).float()

    def randn(self, shape):
        return torch.randn(shape, device=self._device)

    def sqrt(self, value):
        return torch.sqrt(value)

    def as_tensor(self, value):
        return torch.tensor(value, device=self._device)

    def diag(self, value):
        return torch.diag(value)
