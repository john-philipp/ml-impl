from src.enums.enums import TensorHandler
from src.tensor.numpy_tensor_handler import _NumpyTensorHandler
from src.tensor.torch_tensor_handler import _TorchTensorHandler


def find_tensor_handler_cls(tensor_handler: TensorHandler):
    return {
        TensorHandler.TORCH: _TorchTensorHandler,
        TensorHandler.NUMPY: _NumpyTensorHandler
    }[tensor_handler]