import logging

from src.args.parsers.enums import DeviceType
from src.checkpoint.checkpoint_handler import CheckpointHandler
from src.config.config import Config
from src.context.learning_context import LearningContext
from src.context.learning_context_config import LearningContextConfig
from src.tensor import find_tensor_handler_cls
from src.tensor.interfaces import ITensorHandler
from src.tensor.tensor_handler_config import TensorHandlerConfig
from stopwatch import Stopwatch


log = logging.getLogger(__name__)


class Trainer:

    def __init__(self, config: Config, log_handler):
        self._config = config
        self._stopwatch = Stopwatch()
        self._checkpoint_handler = CheckpointHandler(log_handler, self._config.tensor_handler)
        tensor_handler_config = TensorHandlerConfig(use_cuda=self._config.device == DeviceType.CUDA)
        tensor_handler_cls = find_tensor_handler_cls(self._config.tensor_handler)
        self._tensor_handler: ITensorHandler = tensor_handler_cls(log, tensor_handler_config)
        learning_context_config = LearningContextConfig(**config.__dict__)
        self._ctx = LearningContext(learning_context_config, self._tensor_handler)

        self._epochs_trained = 0

    def train(self):
        new_data = False
        epoch = 0
        cost = None

        try:
            if self._config.use_checkpoint:
                self._epochs_trained = self._checkpoint_handler.load_latest(
                    self._ctx.image_shape, self._ctx.try_load_checkpoint)

            log.info("Training...")
            for label, dataset_path in enumerate(self._config.datasets):
                self._ctx.load_data(dataset_path, label)
            self._ctx.accumulate_data()
            self._ctx.normalise_data()

            self._stopwatch.start()
            for epoch in range(self._epochs_trained, self._config.epochs + self._epochs_trained):
                cost = self._ctx.train_epoch()
                new_data = True
                if epoch % self._config.log_every == 0:
                    log.info(f"Epoch {epoch} done after {self._stopwatch.elapsed():.3f}s: cost={cost:.3e}")
                if self._tensor_handler.is_nan(cost):
                    log.error(f"Cost is nan. Quitting. Try again with lower learning rate.")
                    new_data = False
                    break
                elif epoch % self._config.checkpoint_epochs == 0:
                    self._checkpoint_handler.save(
                        epoch, self._ctx.image_shape, cost, self._ctx.save_checkpoint)
                    new_data = False
                self._epochs_trained += 1
        except KeyboardInterrupt:
            self._stopwatch.stop()
            log.info(f"Epoch {epoch} done after {self._stopwatch.elapsed():.3f}s: cost={cost:.3e}")

        if new_data and not self._tensor_handler.is_nan(cost):
            self._checkpoint_handler.save(
                self._epochs_trained, self._ctx.image_shape, cost, self._ctx.save_checkpoint)
