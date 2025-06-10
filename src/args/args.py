import argparse
import logging

from arg_parse.arg_parser import ArgParser
from arg_parse.ifaces import Args
from arg_parse.parser_def_main import ParserDefMain

from src.args.parsers.misc.parser_def_misc import ParserDefMisc
from src.args.parsers.model.parser_def_model import ParserDefModel


log = logging.getLogger(__name__)


class AppArgs(Args):
    def __init__(self):
        super().__init__(globals())
        self.hidden_layer_size = None
        self.checkpoint_epochs = None
        self.use_checkpoint = None
        self.tensor_handler = None
        self.learning_rate = None
        self.batch_offset = None
        self.batch_size = None
        self.log_every = None
        self.datasets = None
        self.points = None
        self.testing = None
        self.device = None
        self.epochs = None
        self.action = None
        self.mode = None
        self.impl = None


def parse_args(*args):
    help_ = False
    if any(x in ["-h", "--help"] for x in args):
        help_ = True

    base_parser = argparse.ArgumentParser(
        prog="logistic-regression",
        description="Simple logistic regression based on gradient descent.")
    base_parser.add_argument("--from-file", default=None, help="Args from file.")
    base_parser.add_argument("--from-env", default=None, help="From env prefix.")

    def register_args():
        parser_def = ParserDefMain()
        parser_def.register_sub_parser(ParserDefModel())
        parser_def.register_sub_parser(ParserDefMisc())
        parser_def.register_args(base_parser)

    if help_:
        register_args()

    base_args = base_parser.parse_known_args(args)
    top_level_args = base_args[0]
    remaining_args = base_args[1]

    register_args()

    arg_parser = ArgParser(
        args_cls=AppArgs,
        from_file_path=top_level_args.from_file,
        from_env_prefix=top_level_args.from_env)

    parsed_args = arg_parser.parse_args(base_parser, *remaining_args)
    parsed_args.log()

    return parsed_args, base_parser
