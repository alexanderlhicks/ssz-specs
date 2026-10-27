"""Check audited import closure using Lean's own import-header parser."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

type Module = tuple[str, ...]

PROJECT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".lake", ".git", ".venv"}


def imports(paths: list[Path]) -> dict[Path, set[Module]]:
    """Parse actual Lean headers, including comments, escaped names and import modifiers."""
    subprocess.run(
        ["lake", "build", "importHeaders"], cwd=PROJECT, capture_output=True, text=True, check=True
    )
    result = subprocess.run(
        [str(PROJECT / ".lake/build/bin/importHeaders"), *(str(path) for path in paths)],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=True,
    )
    parsed = {}
    for line in result.stdout.splitlines():
        path, names = json.loads(line)
        parsed[Path(path)] = {tuple(name) for name in names}
    if set(parsed) != set(paths):
        raise ValueError("Lean header parser did not return every requested source file")
    return parsed


def closure(start: Module, graph: dict[Module, set[Module]]) -> set[Module]:
    """Find imported modules within the audited source tree."""
    seen: set[Module] = set()
    pending = [start]
    while pending:
        module = pending.pop()
        if module not in seen and module in graph:
            seen.add(module)
            pending.extend(graph[module])
    return seen


def check(root: Path) -> list[str]:
    """Report source modules omitted from the proof root or proof imports in implementation."""
    prefixes = ["Ssz"]
    if (root / "SszNative").exists() or (root / "SszNative.lean").exists():
        prefixes.append("SszNative")
    # Shared local modules are bridge nodes, even outside either audited tree.
    # Prune build/dependency environments before walking rather than parsing caches.
    modules = {}
    for directory, subdirs, files in root.walk():
        if directory == root:
            subdirs[:] = [name for name in subdirs if name not in EXCLUDED_DIRS]
            files = [name for name in files if name not in EXCLUDED_DIRS]
        # Path.walk lists directory symlinks among files when follow_symlinks is false.
        # Reject aliases rather than silently omitting a namespace from the graph.
        for name in [*subdirs, *files]:
            path = directory / name
            if path.is_symlink() and path.is_dir():
                raise ValueError(f"source-directory symlinks are unsupported: {path}")
        for filename in files:
            path = directory / filename
            if path.suffix == ".lean":
                modules[path.relative_to(root).with_suffix("").parts] = path
    for prefix in prefixes:
        modules[(prefix,)] = root / f"{prefix}.lean"
    parsed = imports(sorted(modules.values()))
    # Traverse the combined graph across both implementation trees and local bridges.
    for path, names in parsed.items():
        for name in names:
            if name and name[0] in EXCLUDED_DIRS:
                raise ValueError(f"{path}: import from excluded root directory: {'.'.join(name)}")
    graph = {name: parsed[path] for name, path in modules.items()}
    failures = []
    for prefix in prefixes:
        owned = {name for name in modules if name[0] == prefix}
        missing = sorted(owned - closure((prefix, "Proofs"), graph))
        if missing:
            failures.append(
                f"{prefix} modules missing from the audited import closure:\n"
                + "\n".join(".".join(name) for name in missing)
            )
    implementation = set().union(*(closure((prefix,), graph) for prefix in prefixes))
    leaked = sorted(
        name
        for name in implementation
        if len(name) >= 2 and name[:2] in {(prefix, "Proofs") for prefix in prefixes}
    )
    if leaked:
        failures.append(
            "proof modules imported by the implementation:\n"
            + "\n".join(".".join(name) for name in leaked)
        )
    return failures


def main() -> None:
    """Check this package, or a source tree supplied for a regression test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=PROJECT)
    args = parser.parse_args()
    try:
        failures = check(args.root.resolve())
    except subprocess.CalledProcessError as error:
        sys.exit(error.stderr or error.stdout or str(error))
    except (OSError, ValueError) as error:
        sys.exit(str(error))
    if failures:
        sys.exit("\n\n".join(failures))


if __name__ == "__main__":
    main()
