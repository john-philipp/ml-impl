from arg_parse.ifaces import IEnum as IBaseEnum


class IEnum(IBaseEnum):
    @classmethod
    def named_choices(cls):
        return {x: y for x, y in cls.__dict__.items() if not x.startswith("_") and not isinstance(y, classmethod)}


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
