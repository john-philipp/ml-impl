from arg_parse.ifaces import IParserDef

from src.args.parsers.enums import ModelActionType


class ParserDefModelInfer(IParserDef):

    def register_args(self, parent_parser):

        parser = parent_parser.add_parser(
            description="Run inference.",
            help="Check model output against (ideally unseen) input.",
            name=ModelActionType.INFER)

        return parser
