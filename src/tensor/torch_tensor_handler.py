import torch
from PIL.Image import Image
from torchvision import transforms

from src.tensor.interfaces import _ITensorHandler
from src.tensor.tensor_handler_config import TensorHandlerConfig


class _TorchTensorHandler(_ITensorHandler):

    def __init__(self, log, config: TensorHandlerConfig):
        super().__init__(log, config)

        # Load CUDA.
        if self._config.use_cuda:
            if not torch.cuda.is_available():
                self._log.error("CUDA unavailable but configured.")
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
        return checkpoint_data[:-1], checkpoint_data[-1].item()
