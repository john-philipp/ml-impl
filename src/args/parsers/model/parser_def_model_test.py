from arg_parse.ifaces import IParserDef

from src.args.parsers.enums import ModelAction


class ParserDefModelTest(IParserDef):

    def register_args(self, parent_parser):

        parser = parent_parser.add_parser(
            description="Run test.",
            help="Check model output against (ideally unseen) input.",
            name=ModelAction.TEST)

        return parser
