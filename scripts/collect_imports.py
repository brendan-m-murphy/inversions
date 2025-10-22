#!/usr/bin/env python3
"""
Collect and hoist imports from jupytext .py files.

Scans a jupytext-generated .py file (using the percent format), collects all
top-level import statements, deduplicates them while preserving order, removes
the original imports, and inserts an "Imports" block near the top of the file
(after any front matter / metadata).

Usage:
    python scripts/collect_imports.py file1.py file2.py ...
    python scripts/collect_imports.py --dry-run file.py
    python scripts/collect_imports.py --backup file.py

Supports:
  --dry-run   : show what would be changed without writing
  --backup    : create .bak backup before modifying
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import List, Tuple


def parse_jupytext_py(content: str) -> Tuple[List[str], List[str], str]:
    """
    Parse a jupytext .py file and extract:
      - frontmatter lines (the metadata block at the top)
      - import statements (as source strings)
      - the rest of the content (everything else)

    Returns: (frontmatter_lines, import_lines, rest_content)
    """
    lines = content.splitlines(keepends=True)

    # Detect frontmatter (starts with '# ---' and ends with '# ---')
    frontmatter_lines = []
    rest_start = 0

    if lines and lines[0].strip() == "# ---":
        # Find the closing '# ---'
        frontmatter_lines.append(lines[0])
        idx = 1
        while idx < len(lines):
            frontmatter_lines.append(lines[idx])
            if lines[idx].strip() == "# ---":
                rest_start = idx + 1
                break
            idx += 1

    # Parse the rest to find imports
    rest_lines = lines[rest_start:]
    rest_content_str = "".join(rest_lines)

    try:
        tree = ast.parse(rest_content_str)
    except SyntaxError:
        # If we can't parse, return everything as-is
        return frontmatter_lines, [], rest_content_str

    # Collect top-level imports
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Get the source for this import
            import_source = ast.get_source_segment(rest_content_str, node)
            if import_source:
                imports.append(import_source)

    # Now rebuild rest_content without the imports
    # We'll do this by line-based removal
    import_lines_set = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Mark these line numbers for removal
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                for line_num in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    import_lines_set.add(line_num)

    # Reconstruct rest content without import lines
    rest_without_imports = []
    for line_num, line in enumerate(rest_lines, start=1):
        if line_num not in import_lines_set:
            rest_without_imports.append(line)

    return frontmatter_lines, imports, "".join(rest_without_imports)


def deduplicate_imports(imports: List[str]) -> List[str]:
    """Deduplicate imports while preserving order."""
    seen = set()
    result = []
    for imp in imports:
        normalized = imp.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def create_imports_block(imports: List[str]) -> str:
    """Create an Imports cell block in jupytext format."""
    if not imports:
        return ""

    lines = ["# %% [markdown]\n", "# ## Imports\n", "\n", "# %%\n"]
    for imp in imports:
        lines.append(imp + "\n")
    lines.append("\n")
    return "".join(lines)


def process_file(path: Path, dry_run: bool = False, backup: bool = False) -> bool:
    """
    Process a single jupytext .py file to collect and hoist imports.

    Returns True if changes were made, False otherwise.
    """
    content = path.read_text(encoding="utf-8")

    frontmatter, imports, rest = parse_jupytext_py(content)

    if not imports:
        print(f"{path}: No imports found")
        return False

    # Deduplicate imports
    unique_imports = deduplicate_imports(imports)

    print(f"{path}: Found {len(imports)} import(s), {len(unique_imports)} unique")

    if dry_run:
        print("  Imports to be hoisted:")
        for imp in unique_imports:
            print(f"    {imp}")
        return True

    # Create backup if requested
    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(content, encoding="utf-8")
        print(f"  Created backup: {backup_path}")

    # Build new content
    new_content_parts = []

    # Add frontmatter
    if frontmatter:
        new_content_parts.append("".join(frontmatter))

    # Add imports block
    imports_block = create_imports_block(unique_imports)
    new_content_parts.append(imports_block)

    # Add rest of content
    new_content_parts.append(rest)

    new_content = "".join(new_content_parts)

    # Write back
    path.write_text(new_content, encoding="utf-8")
    print(f"  Updated {path}")

    return True


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect and hoist imports in jupytext .py files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="+", type=Path, help="Jupytext .py files to process")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    parser.add_argument("--backup", action="store_true", help="Create .bak backup before modifying")

    args = parser.parse_args(argv)

    any_changes = False
    for file_path in args.files:
        if not file_path.exists():
            print(f"Error: {file_path} does not exist", file=sys.stderr)
            continue

        if file_path.suffix != ".py":
            print(f"Warning: {file_path} is not a .py file, skipping", file=sys.stderr)
            continue

        try:
            changed = process_file(file_path, dry_run=args.dry_run, backup=args.backup)
            any_changes = any_changes or changed
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)

    return 0 if any_changes or args.dry_run else 1


if __name__ == "__main__":
    sys.exit(main())
