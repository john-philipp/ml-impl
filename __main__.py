import logging
import sys
import time

import os

from PIL import ImageOps, Image
from logging import basicConfig, getLogger

from src.args.args import parse_args, Args
from src.checkpoint.checkpoint_handler import CheckpointHandler
from src.context.learning_context_config import LearningContextConfig
from src.enums.enums import Device, TensorHandler, Mode
from src.log.log_handler.log_handler import LogHandler
from src.tensor import find_tensor_handler_cls
from src.tensor.interfaces import _ITensorHandler
from src.tensor.tensor_handler_config import TensorHandlerConfig

basicConfig(level=logging.INFO)
log = getLogger(__name__)


class Context:

    def __init__(self, config: LearningContextConfig, tensor_handler: _ITensorHandler):
        self._config = config
        self.alpha = config.learning_rate  # Learning rate.
        self.dimensions = config.image_size[0] * config.image_size[1] * 3
        self.datas = []

        self.image_shape = config.image_size
        self.image_width = config.image_size[0]
        self.image_height = config.image_size[1]
        self.batch_size = config.batch_size

        self.tensor_handler = tensor_handler

        self.w = self.tensor_handler.zeros(self.dimensions)
        self.b = 0

        self.m = 0  # Sample count.
        self.a = None  # Predictions. Activation function (sigmoid): 1 / (1 + e^-z)
        self.x = None  # Inputs.
        self.y = None  # Truth.

        self.j = None  # Average cost over all samples.
        self.z = None  # Logit (raw score prior to activation: z = w.T dot x + b

        # Derivatives of J (cost) wrt to variable.
        self.dl_dz = None
        self.dw = None
        self.db = None

    @staticmethod
    def load_image(image_path):
        return Image.open(image_path)

    @staticmethod
    def resize_image(image, width, height, padding_rgb=(0, 0, 0)):
        image.thumbnail((width, height))
        delta_w = width - image.width
        delta_h = height - image.height
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        return ImageOps.expand(image, padding, fill=padding_rgb)

    def set_image_size(self, width, height):
        self.image_width = width
        self.image_height = height

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def load_image_data(self, file_path):
        image = self.load_image(file_path)
        image = self.resize_image(image, self.image_width, self.image_height)
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
                self.x[:, offset + j] = x[:, j]
                self.y[offset + j] = label
            offset += m

    def normalise_data(self):
        self.x = self.tensor_handler.normalise(self.x)

    def save_checkpoint(self, path):
        log.info("Trying to save checkpoint...")
        log.info(f"Saving checkpoint: {path}")
        checkpoint_data = self.tensor_handler.concatenate(self.w, self.tensor_handler.scalar_to_array(self.b), axis=0)
        self.tensor_handler.save(checkpoint_data, path)
        log.info("Done saving checkpoint.")

    def try_load_checkpoint(self, path):
        log.info("Trying to load checkpoint...")
        if not path or not os.path.isfile(path):
            log.warning(f"No such checkpoint: {path}")
            return

        log.info(f"Loading checkpoint: {path}")
        checkpoint_data = self.tensor_handler.load(path)
        self.w, self.b = self.tensor_handler.unpack_checkpoint(checkpoint_data)
        log.info("Done loading checkpoint.")

    def update_z(self):
        self.z = self.tensor_handler.multiply(self.w, self.x) + self.b

    def update_a(self):
        self.a = 1 / (1 + self.tensor_handler.exp(-self.z))

    def update_j(self):
        self.j = (-1 / self.m) * (
            self.tensor_handler.multiply(
                self.y,
                self.tensor_handler.log(self.a)) + self.tensor_handler.multiply(
                    1 - self.y, self.tensor_handler.log(1 - self.a)))

    def update_dl_dz(self):
        self.dl_dz = self.a - self.y

    def update_dj_dw(self):
        self.dw = (1 / self.m) * self.tensor_handler.multiply(self.x, self.dl_dz)

    def update_dj_db(self):
        self.db = (1 / self.m) * self.tensor_handler.sum(self.dl_dz)

    def update_w(self):
        self.w -= self.alpha * self.dw

    def update_b(self):
        self.b -= self.alpha * self.db

    def train_epoch(self):
        # Forward propagation.
        self.update_z()
        self.update_a()
        self.update_j()

        # Backward propagation.
        self.update_dl_dz()
        self.update_dj_db()
        self.update_dj_dw()

        # Update weights and bias.
        self.update_w()
        self.update_b()

        return self.j

    def infer(self, image_path):
        self.m = 1
        self.x = self.tensor_handler.zeros((self.dimensions, self.m))
        image_data = self.load_image_data(image_path)

        self.x[:, 0] = image_data
        self.normalise_data()

        self.update_z()
        self.update_a()

        return self.a


if __name__ == '__main__':
    log.info("Starting...")
    args: Args = parse_args(*sys.argv[1:])

    # Validation.
    if args.device == Device.CUDA and args.tensor_handler != TensorHandler.TORCH:
        raise ValueError("Need torch for CUDA.")

    log_handler = LogHandler()
    checkpoint_handler = CheckpointHandler(log_handler, args.tensor_handler)
    tensor_handler_config = TensorHandlerConfig(use_cuda=args.device == Device.CUDA)
    learning_context_config = LearningContextConfig(
        image_size=(args.pts_sqrt, args.pts_sqrt),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size)

    tensor_handler_cls = find_tensor_handler_cls(args.tensor_handler)
    tensor_handler = tensor_handler_cls(log, tensor_handler_config)

    checkpoint_path = "checkpoint" + (".pt" if args.device == Device.CUDA else ".npy")

    ctx = Context(learning_context_config, tensor_handler)
    if args.mode == Mode.TRAIN:
        ctx.load_data(".datasets/cats_dogs/dogs", 0)
        ctx.load_data(".datasets/cats_dogs/cats", 1)
        ctx.accumulate_data()
        ctx.normalise_data()

    if args.use_checkpoint:
        checkpoint_handler.load_latest(ctx.image_shape, ctx.try_load_checkpoint)

    new_data = False
    t0_total = time.time()
    t0 = t0_total
    epoch = 0
    try:
        if args.mode == Mode.TRAIN:

            log.info("Training...")
            for epoch in range(args.epochs):
                epoch += 1
                cost = ctx.train_epoch()
                t1 = time.time()
                new_data = True
                if epoch % args.log_every == 0:
                    log.info(f"Epoch {epoch} done after {t1 - t0:.3f}s: cost={cost:.4f}")
                    t0 = t1
                if epoch % args.checkpoint_epochs == 0:
                    checkpoint_handler.save(epoch, ctx.image_shape, ctx.save_checkpoint)
                    new_data = False

        elif args.mode == Mode.INFER:

            # Infer.
            log.info("Inferring...")
            for type_ in ["dog", "cat"]:
                dir_path = f".datasets/cats_dogs/{type_}s/"
                files = os.listdir(dir_path)
                files.sort()

                # Temporary only. But basically infer against files not in the configured batch.
                for file in files[args.batch_size:args.batch_size + 10]:
                    print(f"{type_}: {ctx.infer(os.path.join(dir_path, file))}")

        else:
            raise ValueError(f"Unknown mode: {args.mode}")

    except KeyboardInterrupt:
        pass

    if new_data:
        checkpoint_handler.save(epoch, ctx.image_shape, ctx.save_checkpoint)

    log.info(f"Done after {time.time() - t0_total:.3f}s.")
