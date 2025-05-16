import logging
import sys
import time

import os

from logging import basicConfig, getLogger

from src.args.args import parse_args, Args
from src.checkpoint.checkpoint_handler import CheckpointHandler
from src.context.learning_context import LearningContext
from src.context.learning_context_config import LearningContextConfig
from src.enums.enums import Device, TensorHandler, Mode
from src.log.log_handler.log_handler import LogHandler
from src.tensor import find_tensor_handler_cls
from src.tensor.tensor_handler_config import TensorHandlerConfig


basicConfig(level=logging.INFO)
log = getLogger(__name__)


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

    ctx = LearningContext(learning_context_config, tensor_handler)
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
