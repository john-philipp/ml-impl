import os
from abc import ABC, abstractmethod
from logging import getLogger

from src.args.parsers.enums import Impl
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

        self.image_shape = (config.points, config.points)
        self.image_width = config.points
        self.image_height = config.points
        self.batch_offset = config.batch_offset
        self.batch_size = config.batch_size

        self.tensor_handler = tensor_handler
        self.m = 0

    def set_image_size(self, width, height):
        self.image_width = width
        self.image_height = height

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def load_image_data(self, file_path):
        image = load_image(file_path)
        image = resize_image(image, self.image_width, self.image_height)
        image_data = self.tensor_handler.image_to_array(image)
        self.tensor_handler.resize_array_1d(image_data, self.dimensions)
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
        x = self.tensor_handler.zeros((self.dimensions, dm))

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
        self.x = self.tensor_handler.zeros((self.dimensions, self.m))
        self.y = self.tensor_handler.zeros(self.m)

        offset = 0
        for label, x in self.datas:
            m = x.shape[1]
            for j in range(0, m):

                if self._config.testing:
                    if label == 0:
                        self.tensor_handler.fill(self.x[:, offset + j], 0)
                    elif label == 1:
                        self.tensor_handler.fill(self.x[:, offset + j], 1)
                    else:
                        raise ValueError()
                else:
                    self.x[:, offset + j] = x[:, j]

                self.y[offset + j] = label
            offset += m

    def normalise_data(self):
        self.x = self.tensor_handler.normalise(self.x)

    def save_checkpoint(self, path):
        if self._config.impl == Impl.NN_RELU:
            log.warning("Not implemented for ReLU")
            return

        log.debug("Trying to save checkpoint...")
        log.info(f"Saving checkpoint: {path}")
        checkpoint_data = self.tensor_handler.concatenate(
            self.w, self.tensor_handler.scalar_to_array(self.b), axis=0)
        self.tensor_handler.save(checkpoint_data, path)
        log.info("Done saving checkpoint.")

    def try_load_checkpoint(self, path):
        log.debug("Trying to load checkpoint...")
        if not path or not os.path.isfile(path):
            log.warning(f"No such checkpoint: {path}")
            return

        log.info(f"Loading checkpoint: {path}")
        checkpoint_data = self.tensor_handler.load(path)
        self.w, self.b = self.tensor_handler.unpack_checkpoint(checkpoint_data)
        log.info("Done loading checkpoint.")

    @abstractmethod
    def train_epoch(self) -> float:
        # Returns cost.
        raise NotImplementedError()

    @abstractmethod
    def infer(self, image_path, expected_label) -> float:
        # Returns prediction.
        raise NotImplementedError()
