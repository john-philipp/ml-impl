import argparse

from arg_parse.parser_main import ParserMain

from src.args.parsers.model.ParserModel import ParserModel


class Args:
    def __init__(self, parsed_args):
        self.checkpoint_epochs = None
        self.use_checkpoint = None
        self.tensor_handler = None
        self.learning_rate = None
        self.batch_offset = None
        self.batch_size = None
        self.log_every = None
        self.datasets = None
        self.pts_sqrt = None
        self.testing = None
        self.device = None
        self.epochs = None
        self.action = None
        self.mode = None

        for property_name in self.__dict__.keys():
            if hasattr(parsed_args, property_name):
                setattr(self, property_name, getattr(parsed_args, property_name))


def parse_args(*args):
    arg_parser = argparse.ArgumentParser(
        prog="logistic-regression",
        description="Simple logistic regression based on gradient descent.")

    parser = ParserMain(arg_parser, [
        ParserModel(),
    ])

    parsed_args = parser.parse_args(*args)
    return Args(parsed_args), arg_parser
