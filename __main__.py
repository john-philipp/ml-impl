import logging
import sys

from logging import basicConfig, getLogger

from src.args.args import parse_args
from src.args.parsers.enums import ModeType, ModelActionType, TensorHandlerType, DeviceType
from src.config.config import Config
from src.log.log_handler.log_handler import LogHandler
from src.trainer.trainer import Trainer
from stopwatch import Stopwatch

basicConfig(level=logging.INFO)
log = getLogger(__name__)


if __name__ == '__main__':
    log.info("Starting...")
    args_ = parse_args(*sys.argv[1:])
    config = Config.from_args(args_)

    # Validation.
    if config.device == DeviceType.CUDA and config.tensor_handler != TensorHandlerType.TORCH:
        raise ValueError("Need torch for CUDA.")

    log_handler = LogHandler()
    stopwatch = Stopwatch()
    stopwatch.start()

    try:
        if args_.mode == ModeType.MODEL:

            if args_.action == ModelActionType.TRAIN:
                trainer = Trainer(config, log_handler)
                trainer.train()

            elif args_.action == ModelActionType.INFER:
                trainer = Trainer(config, log_handler)
                trainer.infer()

            else:
                raise ValueError(f"Unknown mode action: {args_.mode}.{args_.action}")
        else:
            raise ValueError(f"Unknown mode: {args_.mode}")
    except KeyboardInterrupt:
        pass

    log.info(f"Done after {stopwatch.stop():.3f}s.")
