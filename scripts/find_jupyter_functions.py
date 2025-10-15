#!/usr/bin/env python3
"""
Scan a directory of Jupyter notebooks for function/class definitions,
group identical definitions, and emit a report with candidates for extraction.

Usage:
    python scripts/find_jupyter_functions.py --notebooks-dir notebooks --out extracted --min-occurrences 1

Requires: pip install nbformat
"""

import argparse
import ast
import hashlib
import os
import textwrap
from collections import defaultdict
from pathlib import Path

import nbformat


def normalize_source(src: str) -> str:
    # Basic normalization: strip leading/trailing blank lines and dedent
    return textwrap.dedent(src).strip()


def node_to_source(node: ast.AST, source: str) -> str:
    # Try to get original source for a node from the cell source (works in Py3.8+)
    try:
        s = ast.get_source_segment(source, node)
        if s:
            return s
    except Exception:
        pass
    # Fallback to ast.unparse (Py3.9+)
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def hash_source(src: str) -> str:
    return hashlib.sha1(src.encode("utf-8")).hexdigest()


def scan_notebook(path: Path):
    nb = nbformat.read(path, as_version=4)
    results = []
    for cell_index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        source = cell.source
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                src = node_to_source(node, source)
                if not src:
                    continue
                norm = normalize_source(src)
                results.append(
                    {
                        "notebook": str(path),
                        "cell_index": cell_index,
                        "name": getattr(node, "name", "<anon>"),
                        "type": "class" if isinstance(node, ast.ClassDef) else "function",
                        "source": norm,
                        "hash": hash_source(norm),
                    }
                )
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--notebooks-dir", "-n", required=True)
    p.add_argument("--out", "-o", default="extracted")
    p.add_argument("--min-occurrences", "-m", type=int, default=1)
    p.add_argument("--write-examples", action="store_true")
    args = p.parse_args()

    notebooks_dir = Path(args.notebooks_dir)
    notebooks = list(notebooks_dir.rglob("*.ipynb"))

    all_defs = []
    for nb in notebooks:
        all_defs.extend(scan_notebook(nb))

    grouped = defaultdict(list)
    for d in all_defs:
        grouped[d["hash"]].append(d)

    outdir = Path(args.out)

    if args.write_examples:
        outdir.mkdir(parents=True, exist_ok=True)

    report_lines = []
    report_lines.append(f"Scanned {len(notebooks)} notebooks, found {len(all_defs)} defs\n")
    candidates = []
    for h, defs in grouped.items():
        count = len(defs)
        name = defs[0]["name"]
        t = defs[0]["type"]
        sample_nb = defs[0]["notebook"]
        report_lines.append(f"- {t} {name!s} — occurrences: {count} — sample: {sample_nb}")
        if count >= args.min_occurrences:
            candidates.append((h, defs))
            if args.write_examples:
                # write example file
                safe_name = f"{t}_{name}_{h[:8]}"
                out_path = outdir / f"{safe_name}.py"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("# extracted from notebooks:\n")
                    for d in defs[:3]:
                        f.write(f"# - {d['notebook']} (cell {d['cell_index']})\n")
                    f.write("\n\n")
                    f.write(defs[0]["source"])
                report_lines.append(f"  -> wrote example to {out_path}")

    summary = "\n".join(report_lines)
    print(summary)
    print(f"\nCandidates (count >= {args.min_occurrences}): {len(candidates)}")
    if candidates:
        print("Run with --write-examples to write sample .py files into", outdir)


if __name__ == "__main__":
    main()
