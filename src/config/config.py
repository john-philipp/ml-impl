from src.args.args import AppArgs
from src.args.parsers.enums import DeviceType, TensorHandlerType


class Config(AppArgs):
    @classmethod
    def from_args(cls, args: AppArgs):
        config = Config()
        config.__dict__.update(args.__dict__)
        return config

    def use_cuda(self):
        return self.device == DeviceType.CUDA

    def use_torch(self):
        return self.tensor_handler != TensorHandlerType.TORCH
