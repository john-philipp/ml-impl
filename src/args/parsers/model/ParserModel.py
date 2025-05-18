from arg_parse.ifaces import IParser

from src.args.parsers.enums import ModeType, MetaType, TensorHandlerType, DeviceType
from src.args.parsers.model.ParserModelInfer import ParserModelInfer
from src.args.parsers.model.ParserModelTrain import ParserModelTrain
from src.context.learning_context_config import DEFAULT_BATCH_SIZE, DEFAULT_POINTS_SQRT


class ParserModel(IParser):

    def add_args(self, parent_parser):

        sub_parsers = [
            ParserModelTrain(),
            ParserModelInfer()
        ]

        parser = parent_parser.add_parser(
            description="Model related functionality.",
            name=ModeType.MODEL,
            help="Model related functionality.")

        action_parser = parser.add_subparsers(
            dest=MetaType.ACTION,
            help="Model related action to take.",
            required=True)

        for sub_parser in sub_parsers:
            sub_parser = sub_parser.add_args(action_parser)
            sub_parser.add_argument(
                "--tensor-handler", "-t",
                choices=TensorHandlerType.choices(),
                default=TensorHandlerType.TORCH)

            sub_parser.add_argument(
                "--device", "-d",
                choices=DeviceType.choices(),
                default=DeviceType.CUDA)

            sub_parser.add_argument(
                "--batch-size", "-b",
                default=DEFAULT_BATCH_SIZE,
                type=int)

            sub_parser.add_argument(
                "--use-checkpoint", "-c",
                action="store_true")

            sub_parser.add_argument(
                "--testing",
                action="store_true")

            sub_parser.add_argument(
                "--pts-sqrt", "-p",
                default=DEFAULT_POINTS_SQRT,
                type=int)


        return parser

