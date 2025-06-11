import os
from logging import getLogger
from typing import Callable

from src.args.parsers.enums import TensorHandler
from src.config.config import Config
from src.log.log_handler.log_handler import LogHandler


log = getLogger(__name__)


CHECKPOINTS_REL_DIR = "checkpoints"


class CheckpointHandler:
    def __init__(self, config: Config, log_handler: LogHandler):
        self._config = config
        self.ext = self.get_ext(config.tensor_handler)
        self.base_dir = os.path.join(log_handler.curr_log_dir, CHECKPOINTS_REL_DIR)
        os.makedirs(self.base_dir, exist_ok=True)
        self.prev_log_dirs = log_handler.prev_log_dirs

    @staticmethod
    def get_ext(tensor_handler):
        if tensor_handler == TensorHandler.TORCH:
            return "pt"
        elif tensor_handler == TensorHandler.NUMPY:
            return "npy"
        raise ValueError(f"Unknown tensor handler: {tensor_handler}")

    def find_latest(self, path=None):
        existing = os.listdir(path or self.base_dir)
        latest = ''
        if existing:
            existing.sort()
            latest = existing[-1]
        return latest

    def find_next(self, path=None):
        latest = int(self.find_latest(path).split("_")[0] or 0)
        return f"{latest + 1:06}"

    def full_path(self, relative_path):
        return os.path.join(self.base_dir, relative_path)

    def save(self, epoch, cost, suffix, cb_save: Callable[[str], None]):
        next_ = self.find_next()
        checkpoint_name = f"{next_}_{epoch:06}_{cost:.2e}_{suffix}.{self.ext}"
        cb_save(self.full_path(checkpoint_name))

    def load_latest(self, suffix, cb_load: Callable[[str], None]):
        prev_checkpoint = None

        if self.prev_log_dirs:
            self.prev_log_dirs.sort(reverse=True)
            for prev_log_dir in self.prev_log_dirs:
                checkpoint_dir = os.path.normpath(
                    os.path.join(self.base_dir, "../..", prev_log_dir, CHECKPOINTS_REL_DIR))
                latest_checkpoint = self.find_latest(checkpoint_dir)
                # Need a checkpoint of same shape and same tensor handler (numpy, torch).
                if latest_checkpoint and latest_checkpoint.endswith(f"{suffix}.{self.ext}"):
                    prev_checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
                    break

        log.info(f"Will load checkpoint: {prev_checkpoint}")
        cb_load(prev_checkpoint)

        try:
            epochs_passed = int(os.path.basename(prev_checkpoint).split("_")[1])
        except TypeError:
            epochs_passed = 0

        return epochs_passed

