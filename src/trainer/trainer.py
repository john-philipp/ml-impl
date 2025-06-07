import logging
import os

from src.checkpoint.checkpoint_handler import CheckpointHandler
from src.config.config import Config
from src.impl import find_impl_cls
from src.tensor import find_tensor_handler_cls
from src.tensor.interfaces import ITensorHandler
from stopwatch import Stopwatch


log = logging.getLogger(__name__)


class Trainer:

    def __init__(self, config: Config, log_handler):
        self._config = config
        self._stopwatch = Stopwatch()
        self._checkpoint_handler = CheckpointHandler(config, log_handler)
        self._tensor_handler_cls = find_tensor_handler_cls(config.tensor_handler)
        self._tensor_handler: ITensorHandler = self._tensor_handler_cls(config)
        self._impl_cls = find_impl_cls(config.impl)
        self._impl = self._impl_cls(config, self._tensor_handler)
        self._epochs_trained = 0

    def train(self):
        if len(self._config.datasets) != 2:
            raise ValueError("Must specify exactly two datasets.")

        new_data = False
        epoch = 0
        cost = None

        try:
            if self._config.use_checkpoint:
                self._epochs_trained = self._checkpoint_handler.load_latest(
                    self._impl.image_shape, self._impl.try_load_checkpoint)

            log.info("Training...")
            for label, dataset_path in enumerate(self._config.datasets):
                self._impl.load_data(os.path.join(dataset_path, "train"), label)
            self._impl.accumulate_data()
            self._impl.normalise_data()

            self._stopwatch.start()
            total_epochs = self._config.epochs + self._epochs_trained
            for epoch in range(self._epochs_trained, total_epochs):
                cost = self._impl.train_epoch()
                new_data = True
                if epoch % self._config.log_every == 0:
                    log.info(f"Epoch {epoch + 1}/{total_epochs} done after {self._stopwatch.lap():.3f}s: cost={cost.item():.3e}")
                if self._tensor_handler.is_nan(cost):
                    log.error(f"Cost is nan. Quitting. Try again with lower learning rate.")
                    new_data = False
                    break
                elif epoch % self._config.checkpoint_epochs == 0:
                    self._checkpoint_handler.save(
                        epoch, self._impl.image_shape, cost.item(), self._impl.save_checkpoint)
                    new_data = False
                self._epochs_trained += 1
        except KeyboardInterrupt:
            log.info(f"Epoch {epoch} done after {self._stopwatch.stop():.3f}s: cost={cost.item():.3e}")

        if new_data and not self._tensor_handler.is_nan(cost):
            self._checkpoint_handler.save(
                self._epochs_trained, self._impl.image_shape, cost.item(), self._impl.save_checkpoint)

    def infer(self):
        config = self._config
        if len(config.datasets) != 2:
            raise ValueError("Must specify exactly two datasets.")

        # Infer.
        log.info("Inferring...")
        total_count, total_passes = 0, 0
        for expected_label, dataset_path in enumerate(config.datasets):
            files = os.listdir(os.path.join(dataset_path, "test"))
            files.sort()

            count, passes = 0, 0
            for count, file in enumerate(files):
                inferred = self._impl.infer(os.path.join(dataset_path, "test", file), expected_label).item()
                inferred_label = 0 if inferred < 0.5 else 1
                as_expected = inferred_label == expected_label
                relation = "==" if as_expected else "!="
                pass_fail = "PASS" if as_expected else "FAIL"
                log.info(
                    f"{dataset_path}: "
                    f"predicted={inferred:.3f} -> {inferred_label} {relation} {expected_label} {pass_fail}")
                if as_expected:
                    passes += 1
            log.info(f"Pass rate: {passes / (count + 1)}")
            total_count += count + 1
            total_passes += passes
        log.info(f"Overall pass rate: {total_passes / total_count:.3f}")
