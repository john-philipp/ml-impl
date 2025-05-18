import argparse
import logging
import sys
import time

import os

from logging import basicConfig, getLogger

from src.args.args import parse_args, Args
from src.args.parsers.enums import ModeType, ModelActionType, TensorHandlerType, DeviceType
from src.checkpoint.checkpoint_handler import CheckpointHandler
from src.context.learning_context import LearningContext
from src.context.learning_context_config import LearningContextConfig
from src.log.log_handler.log_handler import LogHandler
from src.tensor import find_tensor_handler_cls
from src.tensor.interfaces import ITensorHandler
from src.tensor.tensor_handler_config import TensorHandlerConfig

basicConfig(level=logging.INFO)
log = getLogger(__name__)


if __name__ == '__main__':
    log.info("Starting...")
    args, arg_parser = parse_args(*sys.argv[1:])
    args = Args(args)

    # Validation.
    if args.device == DeviceType.CUDA and args.tensor_handler != TensorHandlerType.TORCH:
        raise ValueError("Need torch for CUDA.")

    log_handler = LogHandler()
    checkpoint_handler = CheckpointHandler(log_handler, args.tensor_handler)
    tensor_handler_config = TensorHandlerConfig(use_cuda=args.device == DeviceType.CUDA)
    learning_context_config = LearningContextConfig(
        image_size=(args.pts_sqrt, args.pts_sqrt),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        testing=args.testing)

    tensor_handler_cls = find_tensor_handler_cls(args.tensor_handler)
    tensor_handler: ITensorHandler = tensor_handler_cls(log, tensor_handler_config)
    ctx = LearningContext(learning_context_config, tensor_handler)

    epoch = 0
    if args.use_checkpoint:
        epoch = checkpoint_handler.load_latest(ctx.image_shape, ctx.try_load_checkpoint)

    new_data = False
    t0_total = time.time()
    t0 = t0_total
    cost = None

    try:
        if args.mode == ModeType.MODEL:

            if args.action == ModelActionType.TRAIN:

                log.info("Training...")
                ctx.load_data(".datasets/cats_dogs/dogs", 0)
                ctx.load_data(".datasets/cats_dogs/cats", 1)
                ctx.accumulate_data()
                ctx.normalise_data()

                for epoch in range(epoch, args.epochs + epoch):
                    epoch += 1
                    cost = ctx.train_epoch()
                    t1 = time.time()
                    new_data = True
                    if epoch % args.log_every == 0:
                        log.info(f"Epoch {epoch} done after {t1 - t0:.3f}s: cost={cost:.3e}")
                        t0 = t1
                    if tensor_handler.is_nan(cost):
                        log.error(f"Cost is nan. Quitting. Try again with lower learning rate.")
                        new_data = False
                        break
                    elif epoch % args.checkpoint_epochs == 0:
                        checkpoint_handler.save(epoch, ctx.image_shape, cost, ctx.save_checkpoint)
                        new_data = False

            elif args.action == ModelActionType.INFER:

                # Infer.
                log.info("Inferring...")
                for type_ in ["dog", "cat"]:
                    dir_path = f".datasets/cats_dogs/{type_}s/"
                    files = os.listdir(dir_path)
                    files.sort()

                    # Temporary only. But basically infer against files not in the configured batch.
                    for file in files[args.batch_size:args.batch_size + 10]:
                        expected_label = 0 if type_ == "dog" else 1
                        print(f"{type_}: predicted={ctx.infer(os.path.join(dir_path, file), expected_label)} expected={expected_label}")
            else:
                raise ValueError(f"Unknown mode action: {args.mode}.{args.action}")
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

    except KeyboardInterrupt:
        pass

    if new_data and not tensor_handler.is_nan(cost):
        checkpoint_handler.save(epoch, ctx.image_shape, cost, ctx.save_checkpoint)

    log.info(f"Done after {time.time() - t0_total:.3f}s.")
