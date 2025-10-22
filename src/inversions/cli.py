#!/usr/bin/env python3
"""
Command-line helpers for the inversions package.

Usage (after installing the package into your uv-managed .venv):
  inversions install_kernel [--venv PATH] [--name NAME] [--display DISPLAY]
                            [--patch-notebooks] [--notebooks PATH] [--force]

This will install an IPython kernel pointing at the specified venv (default:
the Python interpreter that runs this command, which is the usual behaviour
when you call `inversions` from your uv environment). It can also optionally
patch notebooks under the notebooks/ directory to use the new kernelspec.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for _ in range(50):
        if (p / "pyproject.toml").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path.cwd().resolve()


def run(cmd: list[str], capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    if capture:
        return subprocess.run(cmd, text=True, capture_output=True, check=check)
    return subprocess.run(cmd, check=check)


def python_for_venv(venv: str | None) -> str:
    if venv:
        p = Path(venv).expanduser().resolve()
        candidate = p / "bin" / "python"
        if not candidate.exists():
            raise FileNotFoundError(f"Python not found in venv path: {candidate}")
        return str(candidate)
    return sys.executable


def ensure_ipykernel_installed(venv_python: str) -> None:
    try:
        run([venv_python, "-m", "pip", "show", "ipykernel"], check=True)
    except subprocess.CalledProcessError:
        print(
            "ipykernel not found in target Python; installing ipykernel and jupyter_client...",
            file=sys.stderr,
        )
        run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        run([venv_python, "-m", "pip", "install", "ipykernel", "jupyter_client"], check=True)


def kernelspec_exists(name: str) -> bool:
    try:
        from jupyter_client.kernelspec import KernelSpecManager

        ksm = KernelSpecManager()
        specs = ksm.find_kernel_specs()
        return name in specs
    except Exception:
        js = shutil.which("jupyter")
        if js:
            try:
                cp = run([js, "kernelspec", "list", "--json"], capture=True, check=True)
                data = json.loads(cp.stdout)
                names = list(data.get("kernelspecs", {}).keys())
                return name in names
            except Exception:
                return False
        return False


def install_kernel(
    venv: str | None,
    name: str,
    display: str,
    patch_notebooks: bool = False,
    notebooks_path: str | None = None,
    force: bool = False,
) -> int:
    repo_root = find_repo_root()
    notebooks_dir = Path(notebooks_path) if notebooks_path else (repo_root / "notebooks")

    venv_python = python_for_venv(venv)
    print(f"Using Python: {venv_python}")

    ensure_ipykernel_installed(venv_python)

    if kernelspec_exists(name) and not force:
        print(f"Kernel '{name}' already exists. Use --force to reinstall/overwrite.")
        return 0

    print(f"Installing kernel '{name}' (display name: {display}) using {venv_python} ...")
    run(
        [venv_python, "-m", "ipykernel", "install", "--user", "--name", name, "--display-name", display],
        check=True,
    )
    print("Kernel installed.")

    if patch_notebooks:
        patched = 0
        try:
            import nbformat  # type: ignore
        except Exception:
            nbformat = None  # type: ignore

        if not notebooks_dir.exists():
            print(f"Notebooks dir {notebooks_dir} does not exist; skipping patch step.")
            return 0

        for nb_path in notebooks_dir.rglob("*.ipynb"):
            try:
                if nbformat is not None:
                    nb = nbformat.read(nb_path, as_version=4)
                    nb.setdefault("metadata", {})["kernelspec"] = {
                        "name": name,
                        "display_name": display,
                        "language": nb.get("metadata", {}).get("kernelspec", {}).get("language", "python"),
                    }
                    nbformat.write(nb, nb_path)
                else:
                    with open(nb_path, "r", encoding="utf8") as fh:
                        data = json.load(fh)
                    data.setdefault("metadata", {})["kernelspec"] = {
                        "name": name,
                        "display_name": display,
                        "language": data.get("metadata", {}).get("kernelspec", {}).get("language", "python"),
                    }
                    with open(nb_path, "w", encoding="utf8") as fh:
                        json.dump(data, fh, indent=1)
                patched += 1
            except Exception as e:
                print(f"Failed to patch {nb_path}: {e}", file=sys.stderr)
        print(f"Patched {patched} notebook(s) under {notebooks_dir}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inversions")
    sub = parser.add_subparsers(dest="command", required=True)

    p_install = sub.add_parser("install_kernel", help="Install an IPython kernel for the project's venv")
    p_install.add_argument(
        "--venv", help="Path to a Python virtualenv (default: the current interpreter / uv .venv)"
    )
    p_install.add_argument("--name", default="inversions", help="Kernel name (default: inversions)")
    p_install.add_argument(
        "--display",
        default="Python (inversions)",
        help='Kernel display name (default: "Python (inversions)")',
    )
    p_install.add_argument(
        "--patch-notebooks",
        action="store_true",
        help="Update notebooks/ .ipynb files to point to the new kernel",
    )
    p_install.add_argument("--notebooks", help="Directory with notebooks (default: ./notebooks)")
    p_install.add_argument(
        "--force", action="store_true", help="Force reinstall even if the kernel already exists"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "install_kernel":
        return install_kernel(
            venv=args.venv,
            name=args.name,
            display=args.display,
            patch_notebooks=args.patch_notebooks,
            notebooks_path=args.notebooks,
            force=args.force,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
