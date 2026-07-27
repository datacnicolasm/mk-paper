"""CLI de mk-paper para pruebas aisladas de componentes."""

from __future__ import annotations

import argparse
import sys

from mk_paper.cli.literature import add_literature_parser, run_literature_search_cli


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser raíz con subcomandos."""
    parser = argparse.ArgumentParser(
        prog="mk-paper",
        description="CLI de mk-paper para probar componentes de forma aislada.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    literature_parser = subparsers.add_parser(
        "literature",
        help=(
            "Revisión sistemática (ResearchBrief + Groq) o búsqueda cruda; "
            "persiste en output/literature/"
        ),
    )
    add_literature_parser(literature_parser)
    literature_parser.set_defaults(handler=run_literature_search_cli)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
