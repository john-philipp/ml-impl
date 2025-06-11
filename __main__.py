import logging
import sys

from logging import basicConfig, getLogger

from arg_parse.make_docs import make_docs

from src.args.args import parse_args
from src.args.parsers.enums import Mode, ModelAction, TensorHandler, Device, MiscAction
from src.config.config import Config
from src.log.log_handler.log_handler import LogHandler
from src.trainer.trainer import Trainer
from stopwatch import Stopwatch

basicConfig(level=logging.INFO)
log = getLogger(__name__)


if __name__ == '__main__':
    log.info("Starting...")
    args_, base_parser = parse_args(*sys.argv[1:])
    config = Config.from_args(args_)

    # Validation.
    if config.device == Device.CUDA and config.tensor_handler != TensorHandler.TORCH:
        raise ValueError("Need torch for CUDA.")

    log_handler = LogHandler()
    stopwatch = Stopwatch()
    stopwatch.start()

    try:
        if args_.mode == Mode.MODEL:
            if args_.action == ModelAction.TRAIN:
                trainer = Trainer(config, log_handler)
                trainer.train()

            elif args_.action == ModelAction.TEST:
                trainer = Trainer(config, log_handler)
                trainer.test()

            else:
                raise ValueError(f"Unknown mode action: {args_.mode}.{args_.action}")

        elif args_.mode == Mode.MISC:
            if args_.action == MiscAction.MAKE_DOCS:
                make_docs(base_parser)

            else:
                raise ValueError(f"Unknown misc action: {args_.mode}.{args_.action}")

        else:
            raise ValueError(f"Unknown mode: {args_.mode}")

    except KeyboardInterrupt:
        pass

    log.info(f"Done after {stopwatch.stop():.3f}s.")
