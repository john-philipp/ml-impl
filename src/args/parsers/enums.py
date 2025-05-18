from arg_parse.ifaces import IEnum


class MetaType(IEnum):
    ACTION = "action"
    MODE = "mode"


class ModeType(IEnum):
    MODEL = "model"


class ModelActionType(IEnum):
    TRAIN = "train"
    INFER = "infer"


class TensorHandlerType(IEnum):
    TORCH = "torch"
    NUMPY = "numpy"


class DeviceType(IEnum):
    CUDA = "cuda"
    CPU = "cpu"
