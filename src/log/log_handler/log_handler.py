import os

from src.methods import get_timestamp


DEFAULT_LOGS_REL_DIR = "logs"


class LogHandler:
    def __init__(self, log_base_dir=DEFAULT_LOGS_REL_DIR, timestamp=None):
        self.base_dir = log_base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.prev_log_dirs = os.listdir(self.base_dir)

        # New log dir for this session.
        self.timestamp = timestamp or get_timestamp()
        self.curr_log_dir = self.full_path(self.timestamp)
        os.makedirs(self.curr_log_dir, exist_ok=True)

    def full_path(self, relative_path):
        return os.path.join(self.base_dir, relative_path)


