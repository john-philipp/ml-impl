import os
import pickle
from abc import ABC, abstractmethod
from logging import getLogger

from src.config.config import Config
from src.methods import load_image, resize_image
from src.tensor.interfaces import ITensorHandler


log = getLogger(__name__)


class IImpl(ABC):

    def __init__(self, config: Config, tensor_handler: ITensorHandler):
        self._config = config
        self.alpha = config.learning_rate  # Learning rate.
        self.dimensions = config.points * config.points * 3
        self.datas = []
        self._X = None  # Inputs.
        self._y = None  # Truth.

        self.image_shape = (config.points, config.points)
        self.image_width = config.points
        self.image_height = config.points
        self.batch_offset = config.batch_offset
        self.batch_size = config.batch_size

        self._th = tensor_handler
        self.m = 0  # Sample count.

    def set_image_size(self, width, height):
        self.image_width = width
        self.image_height = height

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def load_image_data(self, file_path):
        image = load_image(file_path)
        image = resize_image(image, self.image_width, self.image_height)
        image_data = self._th.image_to_array(image)
        self._th.resize_array_1d(image_data, self.dimensions)
        return image_data

    def load_data(self, path, label):
        files = os.listdir(path)
        files.sort()

        dm = len(files)
        if self.batch_size and self.batch_size < dm:
            files = files[:self.batch_size]
            dm = self.batch_size
        self.m += dm

        # Note, without batching this will eventually incur out of memory errors.
        x = self._th.zeros((self.dimensions, dm))

        i = 0
        for file in files:
            file_path = os.path.join(path, file)
            image_data = self.load_image_data(file_path)
            x[:, i] = image_data
            i += 1

        self.datas.append((label, x))

    def accumulate_data(self):
        assert self.datas, "Must load data prior."
        assert self.m == sum([x.shape[1] for _, x in self.datas]), "Failed consistency check!"

        # Note, without batching this will eventually incur out of memory errors.
        self._X = self._th.zeros((self.dimensions, self.m))
        self._y = self._th.zeros(self.m)

        offset = 0
        for label, x in self.datas:
            m = x.shape[1]
            for j in range(0, m):

                if self._config.testing:
                    if label == 0:
                        self._th.fill(self._X[:, offset + j], 0)
                    elif label == 1:
                        self._th.fill(self._X[:, offset + j], 1)
                    else:
                        raise ValueError()
                else:
                    self._X[:, offset + j] = x[:, j]

                self._y[offset + j] = label
            offset += m

    def normalise_data(self):
        self._X = self._th.normalise(self._X)

    def save_checkpoint(self, path):
        log.debug("Trying to save checkpoint...")
        log.info(f"Saving checkpoint: {path}")
        checkpoint_data = self.get_state()
        self._save(checkpoint_data, path)
        log.info("Done saving checkpoint.")

    def try_load_checkpoint(self, path):
        log.debug("Trying to load checkpoint...")
        if not path or not os.path.isfile(path):
            log.warning(f"No such checkpoint: {path}")
            return

        log.info(f"Loading checkpoint: {path}")
        checkpoint_data = self._load(path)
        self.from_state(checkpoint_data)
        log.info("Done loading checkpoint.")

    @abstractmethod
    def train_epoch(self) -> float:
        # Returns cost.
        raise NotImplementedError()

    @abstractmethod
    def infer(self, image_path, expected_label) -> float:
        # Returns prediction.
        raise NotImplementedError()

    @abstractmethod
    def get_state(self):
        raise NotImplementedError()

    @abstractmethod
    def from_state(self, state):
        raise NotImplementedError()

    def get_checkpoint_suffix(self):
        points = self._config.points
        impl_name = self._config.impl
        return f"{impl_name}_p{points}"

    @staticmethod
    def _save(data, path):
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def _load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
