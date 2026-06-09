from __future__ import annotations

import argparse
from pathlib import Path

from ingestion.loaders import SUPPORTED_EXTENSIONS
from pipeline.run_pipeline import run_pipeline


KNOWN_REGULATION_ORDER = {
    "hipaa.pdf": 0,
    "eu_ai_act.pdf": 1,
    "gdpr.pdf": 2,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local-first regulation-to-Neo4j ontology extraction pipeline."
    )
    parser.add_argument("document_path", nargs="+", help="Path to one or more .pdf, .txt, or .md regulation documents")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        input_paths = resolve_input_paths(args.document_path)
    except ValueError as exc:
        parser.error(str(exc))
    run_pipeline(input_paths)


def resolve_input_paths(inputs: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for value in inputs:
        path = Path(value).expanduser()
        if not path.exists():
            raise ValueError(f"Input path not found: {path}")
        if path.is_dir():
            files = _supported_files_in_directory(path)
            if not files:
                raise ValueError(
                    f"No supported input files found in {path}. "
                    f"Supported extensions: {_format_supported_extensions()}"
                )
            resolved.extend(files)
            continue
        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported input file: {path}. "
                    f"Supported extensions: {_format_supported_extensions()}"
                )
            resolved.append(path)
            continue
        raise ValueError(f"Input path is not a file or directory: {path}")
    return resolved


def _supported_files_in_directory(path: Path) -> list[Path]:
    files = [child for child in path.iterdir() if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files, key=_document_sort_key)


def _document_sort_key(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    return (KNOWN_REGULATION_ORDER.get(name, len(KNOWN_REGULATION_ORDER)), name)


def _format_supported_extensions() -> str:
    return ", ".join(sorted(SUPPORTED_EXTENSIONS))


if __name__ == "__main__":
    main()
