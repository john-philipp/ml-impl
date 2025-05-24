import argparse
import logging

from arg_parse.arg_parser import ArgParser
from arg_parse.ifaces import Args
from arg_parse.parser_def_main import ParserDefMain

from src.args.parsers.model.parser_def_model import ParserDefModel


log = logging.getLogger(__name__)


def get_args_path():
    args_path = "_args/args.yml"
    return args_path


def get_env_var_prefix():
    return "SAMPLE"


class AppArgs(Args):
    def __init__(self):
        super().__init__(globals())
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


def parse_args(*args):
    base_parser = argparse.ArgumentParser(
        prog="logistic-regression",
        description="Simple logistic regression based on gradient descent.")

    parser_def = ParserDefMain()
    parser_def.register_sub_parser(ParserDefModel())
    parser_def.register_args(base_parser)

    arg_parser = ArgParser(
        args_cls=AppArgs,
        from_file_path=get_args_path(),
        from_env_prefix=get_env_var_prefix())

    parsed_args = arg_parser.parse_args(base_parser, *args)
    parsed_args.log()

    return parsed_args
