from src.args.parsers.enums import TensorHandlerType
from src.tensor.numpy_tensor_handler import _NumpyTensorHandler
from src.tensor.torch_tensor_handler import _TorchTensorHandler


def find_tensor_handler_cls(tensor_handler: TensorHandlerType):
    return {
        TensorHandlerType.TORCH: _TorchTensorHandler,
        TensorHandlerType.NUMPY: _NumpyTensorHandler
    }[tensor_handler]
