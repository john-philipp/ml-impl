class LearningContextConfig:
    def __init__(
            self, points: int, learning_rate: float,
            batch_offset: int, batch_size: int, testing: bool, **_):

        self.learning_rate: float = learning_rate
        self.image_size: tuple[int, int] = (points, points)
        self.batch_offset: int = batch_offset
        self.batch_size: int = batch_size
        self.testing: bool = testing
