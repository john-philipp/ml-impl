import argparse

from src.context.learning_context_config import DEFAULT_LEARNING_RATE, DEFAULT_POINTS_SQRT, DEFAULT_BATCH_SIZE, \
    DEFAULT_EPOCHS, DEFAULT_LOG_EVERY
from src.enums.enums import TensorHandler, Device, Mode


class Args:
    def __init__(self, parsed_args):
        self.tensor_handler = parsed_args.tensor_handler
        self.device = parsed_args.device
        self.learning_rate = parsed_args.learning_rate
        self.pts_sqrt = parsed_args.pts_sqrt
        self.batch_size = parsed_args.batch_size
        self.epochs = parsed_args.epochs
        self.log_every = parsed_args.log_every
        self.mode = parsed_args.mode
        self.use_checkpoint = parsed_args.use_checkpoint


def parse_args(*args):
    arg_parser = argparse.ArgumentParser(
        prog="logistic-regression",
        description="Simple logistic regression based on gradient descent.")

    arg_parser.add_argument(
        "--tensor-handler", "-t",
        choices=TensorHandler.choices(),
        default=TensorHandler.TORCH)

    arg_parser.add_argument(
        "--device", "-d",
        choices=Device.choices(),
        default=Device.CUDA)

    arg_parser.add_argument(
        "--learning-rate", "-l",
        default=DEFAULT_LEARNING_RATE,
        type=float)

    arg_parser.add_argument(
        "--pts-sqrt", "-p",
        default=DEFAULT_POINTS_SQRT,
        type=int)

    arg_parser.add_argument(
        "--batch-size", "-b",
        default=DEFAULT_BATCH_SIZE,
        type=int)

    arg_parser.add_argument(
        "--epochs", "-e",
        default=DEFAULT_EPOCHS,
        type=int)

    arg_parser.add_argument(
        "--log-every",
        default=DEFAULT_LOG_EVERY,
        type=int)

    arg_parser.add_argument(
        "--mode", "-m",
        default=Mode.TRAIN,
        choices=Mode.choices())

    arg_parser.add_argument(
        "--use-checkpoint", "-c",
        action="store_true")

    parsed_args = arg_parser.parse_args(args=args)
    return Args(parsed_args)
