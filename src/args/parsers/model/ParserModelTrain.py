from arg_parse.ifaces import IParser

from src.args.parsers.enums import ModelActionType
from src.context.learning_context_config import DEFAULT_LEARNING_RATE, \
    DEFAULT_EPOCHS, DEFAULT_LOG_EVERY, DEFAULT_CHECKPOINT_EPOCHS


class ParserModelTrain(IParser):

    def add_args(self, parent_parser):

        parser = parent_parser.add_parser(
            description="Train a model.",
            name=ModelActionType.TRAIN)

        parser.add_argument(
            "--learning-rate", "-l",
            default=DEFAULT_LEARNING_RATE,
            type=float)

        parser.add_argument(
            "--epochs", "-e",
            default=DEFAULT_EPOCHS,
            type=int)

        parser.add_argument(
            "--log-every",
            default=DEFAULT_LOG_EVERY,
            type=int)

        parser.add_argument(
            "--checkpoint-epochs",
            default=DEFAULT_CHECKPOINT_EPOCHS,
            type=int)

        return parser
