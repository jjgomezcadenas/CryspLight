#!/usr/bin/env python3
"""Compile one CryspLight LaTeX source file and remove build artefacts.

Usage:
    python3 latex/compile_tex.py crysplight_e2e.tex
    python3 latex/compile_tex.py latex/crysplight_e2e
    python3 latex/compile_tex.py docs_fullsim/plan/working_plan.tex
    python3 latex/compile_tex.py --clean crysplight_e2e.tex

Sources may live in any directory of the repository (latex/, docs_fullsim/, ...);
a bare name is looked up in the current directory and then in latex/.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


LATEX_DIR = Path(__file__).resolve().parent
REPO_DIR = LATEX_DIR.parent
BUILD_SUFFIXES = (
    ".aux",
    ".bbl",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".nav",
    ".out",
    ".run.xml",
    ".snm",
    ".synctex.gz",
    ".toc",
    ".vrb",
)


def source_path(name: str) -> Path:
    """Resolve a source name and ensure it is a .tex file inside the repository."""
    path = Path(name)
    if path.suffix == "":
        path = path.with_suffix(".tex")
    if path.suffix != ".tex":
        raise ValueError("the input file must have a .tex extension")

    if path.is_absolute():
        path = path.resolve()
    else:
        # Accept both ``latex/note.tex`` from the repository root and
        # ``note.tex`` from either the repository root or latex/ itself.
        from_cwd = (Path.cwd() / path).resolve()
        path = from_cwd if from_cwd.exists() else (LATEX_DIR / path).resolve()
    try:
        path.relative_to(REPO_DIR)
    except ValueError as error:
        raise ValueError(f"source must be below {REPO_DIR}") from error
    if not path.is_file():
        raise ValueError(f"LaTeX source not found: {path}")
    return path


def clean(source: Path) -> None:
    """Remove compiler by-products, preserving the source and final PDF."""
    stem = source.with_suffix("")
    for suffix in BUILD_SUFFIXES:
        artefact = Path(f"{stem}{suffix}")
        if artefact.exists():
            artefact.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile a LaTeX source fully, then remove build artefacts."
    )
    parser.add_argument("source", help=".tex file to compile (relative to the repository or latex/)")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove LaTeX build artefacts only; do not compile",
    )
    args = parser.parse_args()

    try:
        source = source_path(args.source)
    except ValueError as error:
        parser.error(str(error))

    if args.clean:
        clean(source)
        print(f"Removed build artefacts for {source.name}")
        return 0

    latexmk = shutil.which("latexmk")
    if latexmk is None:
        parser.error("latexmk is required but was not found on PATH")

    # latexmk runs as many LaTeX/BibTeX/Biber passes as needed (at least two when
    # cross-references require it), while leaving the PDF alongside the source.
    command = [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", source.name]
    try:
        result = subprocess.run(command, cwd=source.parent).returncode
    finally:
        clean(source)

    if result == 0:
        print(f"Compiled {source.name} -> {source.with_suffix('.pdf').name}")
    return result


if __name__ == "__main__":
    sys.exit(main())
