from __future__ import annotations

import argparse

from pipeline.run_pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local-first regulation-to-Neo4j ontology extraction pipeline."
    )
    parser.add_argument("document_path", help="Path to a .pdf, .txt, or .md regulation document")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args.document_path)


if __name__ == "__main__":
    main()
