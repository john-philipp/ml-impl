import logging
import sys
import time

import os

from logging import basicConfig, getLogger

from src.args.args import parse_args, Args, AppArgs
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
    args = parse_args(*sys.argv[1:])

    # Validation.
    if args.device == DeviceType.CUDA and args.tensor_handler != TensorHandlerType.TORCH:
        raise ValueError("Need torch for CUDA.")

    log_handler = LogHandler()
    checkpoint_handler = CheckpointHandler(log_handler, args.tensor_handler)
    tensor_handler_config = TensorHandlerConfig(use_cuda=args.device == DeviceType.CUDA)
    learning_context_config = LearningContextConfig(**args.__dict__)

    tensor_handler_cls = find_tensor_handler_cls(args.tensor_handler)
    tensor_handler: ITensorHandler = tensor_handler_cls(log, tensor_handler_config)
    ctx = LearningContext(learning_context_config, tensor_handler)

    new_data = False
    t0_total = time.time()
    t0 = t0_total
    cost = None

    try:
        if args.mode == ModeType.MODEL:

            if args.action == ModelActionType.TRAIN:
                if len(args.datasets) != 2:
                    raise ValueError("Must specify exactly two datasets.")

                new_data = train(args, t0)

            elif args.action == ModelActionType.INFER:
                if len(args.datasets) != 2:
                    raise ValueError("Must specify exactly two datasets.")

                # Infer.
                log.info("Inferring...")
                total_count, total_passes = 0, 0
                for expected_label, dataset_path in enumerate(args.datasets):
                    files = os.listdir(dataset_path)
                    files.sort()

                    count, passes = 0, 0
                    for count, file in enumerate(files[args.batch_offset: args.batch_offset + args.batch_size]):
                        inferred = ctx.infer(os.path.join(dataset_path, file), expected_label).item()
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
            else:
                raise ValueError(f"Unknown mode action: {args.mode}.{args.action}")
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

    except KeyboardInterrupt:
        pass

    if new_data and not tensor_handler.is_nan(cost):
        checkpoint_handler.save(epoch, ctx.image_shape, cost, ctx.save_checkpoint)

    log.info(f"Done after {time.time() - t0_total:.3f}s.")
