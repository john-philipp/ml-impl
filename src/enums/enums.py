from src.enums.interfaces import _IEnum


class TensorHandler(_IEnum):
    TORCH = "torch"
    NUMPY = "numpy"


class Device(_IEnum):
    CUDA = "cuda"
    CPU = "cpu"


class Mode(_IEnum):
    TRAIN = "train"
    INFER = "infer"
