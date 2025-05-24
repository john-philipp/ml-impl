from arg_parse.ifaces import IParserDef

from src.args.defaults import Defaults
from src.args.parsers.enums import ModelActionType


class ParserDefModelTrain(IParserDef):

    def register_args(self, parent_parser):

        parser = parent_parser.add_parser(
            description="Train a model.",
            name=ModelActionType.TRAIN)

        parser.add_argument(
            "--learning-rate", "-l",
            help="Specify the learning rate of the model.",
            default=Defaults.learning_rate,
            type=float)

        parser.add_argument(
            "--epochs", "-e",
            help="Run for this many epochs.",
            default=Defaults.epochs,
            type=int)

        parser.add_argument(
            "--log-every",
            help="Logs cost every {log-every} epochs.",
            default=Defaults.log_every,
            type=int)

        parser.add_argument(
            "--checkpoint-epochs",
            help="Create a checkpoint every this many epochs.",
            default=Defaults.checkpoint_epochs,
            type=int)

        return parser
