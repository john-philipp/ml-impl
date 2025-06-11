from arg_parse.ifaces import IParserDef

from src.args.parsers.enums import Mode, Meta
from src.args.parsers.misc.parser_def_misc_make_docs import ParserDefMiscMakeDocs


class ParserDefMisc(IParserDef):

    def register_args(self, parent_parser):

        sub_parsers = [
            ParserDefMiscMakeDocs()
        ]

        parser = parent_parser.add_parser(
            description="Misc functionality.",
            name=Mode.MISC,
            help="Misc functionality.")

        action_parser = parser.add_subparsers(
            dest=Meta.ACTION,
            help="Misc action to take.",
            required=True)

        for sub_parser in sub_parsers:
            sub_parser.register_args(action_parser)

        return parser

