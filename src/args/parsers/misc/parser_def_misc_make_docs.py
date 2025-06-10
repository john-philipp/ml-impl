from arg_parse.ifaces import IParserDef

from src.args.parsers.enums import MiscAction


class ParserDefMiscMakeDocs(IParserDef):

    def register_args(self, parent_parser):

        parser = parent_parser.add_parser(
            description="Make docs.",
            help="Make docs.",
            name=MiscAction.MAKE_DOCS)

        return parser
