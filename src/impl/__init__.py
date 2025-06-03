from src.args.parsers.enums import Impl
from src.impl.interfaces import IImpl
from src.impl.log_reg_impl import _LogRegImpl
from src.impl.nn_relu_impl import _NnReluImpl


def find_impl_cls(impl: Impl) -> IImpl.__class__:
    return {
        Impl.LOG_REG: _LogRegImpl,
        Impl.NN_RELU: _NnReluImpl,
    }[impl]
