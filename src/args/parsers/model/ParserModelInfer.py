from arg_parse.ifaces import IParser

from src.args.parsers.enums import ModelActionType


class ParserModelInfer(IParser):

    def add_args(self, parent_parser):

        parser = parent_parser.add_parser(
            description="Run inference.",
            name=ModelActionType.INFER)

        return parser
