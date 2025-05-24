from arg_parse.ifaces import IParserDef

from src.args.defaults import Defaults
from src.args.parsers.enums import ModeType, MetaType, TensorHandlerType, DeviceType
from src.args.parsers.model.parser_def_model_infer import ParserDefModelInfer
from src.args.parsers.model.parser_def_model_train import ParserDefModelTrain


class ParserDefModel(IParserDef):

    def register_args(self, parent_parser):

        sub_parsers = [
            ParserDefModelTrain(),
            ParserDefModelInfer()
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
            sub_parser = sub_parser.register_args(action_parser)
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
                default=Defaults.batch_offset,
                type=int)

            sub_parser.add_argument(
                "--batch-size", "-b",
                help="Use this many files in each dataset directory (if available).",
                default=Defaults.batch_size,
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
                "--points", "-p",
                help="Determines the size of picture actually learnt on. "
                     "Aspect ratio is preserved. Pictures are padded.",
                default=Defaults.points,
                type=int)

            sub_parser.add_argument(
                "--datasets", "-s",
                help="Dataset folders, labels are set by position in the list. Require exactly two arguments.",
                default=[".datasets/cats_dogs/dogs", ".datasets/cats_dogs/cats"],
                nargs="+")

        return parser

