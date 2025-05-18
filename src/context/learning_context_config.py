DEFAULT_POINTS_SQRT = 64
DEFAULT_IMAGE_SIZE = (DEFAULT_POINTS_SQRT, DEFAULT_POINTS_SQRT)
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 100
DEFAULT_EPOCHS = 100
DEFAULT_LOG_EVERY = 1000
DEFAULT_CHECKPOINT_EPOCHS = 10000
DEFAULT_TESTING = False


class LearningContextConfig:
    def __init__(
            self, image_size: tuple[int, int] = None, learning_rate: float = None,
            batch_size: int = None, testing: bool = None):

        self.learning_rate: float = learning_rate or DEFAULT_LEARNING_RATE
        self.image_size: tuple[int, int] = image_size or DEFAULT_IMAGE_SIZE
        self.batch_size: int = batch_size or DEFAULT_BATCH_SIZE
        self.testing: bool = testing or DEFAULT_TESTING
