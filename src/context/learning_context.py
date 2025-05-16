from src.context.learning_context_config import LearningContextConfig


class LearningContext:
    def __init__(self, config: LearningContextConfig):
        self._config = config

        # Dimensions are WxHx3.
        self.dimensions = config.image_size[0] * config.image_size[1] * 3

        self.datas = []