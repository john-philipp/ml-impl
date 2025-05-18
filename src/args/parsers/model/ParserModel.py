from arg_parse.ifaces import IParser

from src.args.parsers.enums import ModeType, MetaType, TensorHandlerType, DeviceType
from src.args.parsers.model.ParserModelInfer import ParserModelInfer
from src.args.parsers.model.ParserModelTrain import ParserModelTrain
from src.context.learning_context_config import DEFAULT_BATCH_SIZE, DEFAULT_POINTS_SQRT, DEFAULT_BATCH_OFFSET


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
                help="Which tensor handler to use (CUDA requires torch).",
                choices=TensorHandlerType.choices(),
                default=TensorHandlerType.TORCH)

            sub_parser.add_argument(
                "--device", "-d",
                help="Run on CPU or CUDA (CUDA requires torch)",
                choices=DeviceType.choices(),
                default=DeviceType.CUDA)

            sub_parser.add_argument(
                "--batch-offset", "-o",
                help="Offset {batch_size} by {base_offset}. "
                     "Allows simple adding of further data into training/inference. "
                     "Basically a simple range to define our batch.",
                default=DEFAULT_BATCH_OFFSET,
                type=int)

            sub_parser.add_argument(
                "--batch-size", "-b",
                help="Use this many files in each dataset directory (if available).",
                default=DEFAULT_BATCH_SIZE,
                type=int)

            sub_parser.add_argument(
                "--use-checkpoint", "-c",
                help="Load latest checkpoint for architecture and dimension.",
                action="store_true")

            sub_parser.add_argument(
                "--testing",
                help="Use simplistic data for testing (sanity check).",
                action="store_true")

            sub_parser.add_argument(
                "--pts-sqrt", "-p",
                help="Determines the size of picture actually learnt on. "
                     "Aspect ratio is preserved. Pictures are padded.",
                default=DEFAULT_POINTS_SQRT,
                type=int)

            sub_parser.add_argument(
                "--datasets", "-s",
                help="Dataset folders, labels are set by position in the list. Require exactly two arguments.",
                default=[".datasets/cats_dogs/dogs", ".datasets/cats_dogs/cats"],
                nargs="+")

        return parser

