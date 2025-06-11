from arg_parse.ifaces import IEnum as IBaseEnum


class IEnum(IBaseEnum):
    @classmethod
    def named_choices(cls):
        return {x: y for x, y in cls.__dict__.items() if not x.startswith("_") and not isinstance(y, classmethod)}


class Meta(IEnum):
    ACTION = "action"
    MODE = "mode"


class Mode(IEnum):
    MODEL = "model"
    MISC = "misc"


class MiscAction(IEnum):
    MAKE_DOCS = "make-docs"


class ModelAction(IEnum):
    TRAIN = "train"
    TEST = "test"


class TensorHandler(IEnum):
    TORCH = "torch"
    NUMPY = "numpy"


class Device(IEnum):
    CUDA = "cuda"
    CPU = "cpu"


class Impl(IEnum):
    LOG_REG = "log-reg"
    NN_RELU = "nn-relu"
