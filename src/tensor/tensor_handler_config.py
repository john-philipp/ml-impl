DEFAULT_USE_CUDA = True


class TensorHandlerConfig:
    def __init__(self, use_cuda: bool = None):
        self.use_cuda: bool = use_cuda if use_cuda is not None else DEFAULT_USE_CUDA
