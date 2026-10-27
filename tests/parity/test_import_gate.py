"""The audited source graph must come from real Lean import headers."""

import subprocess
import sys
from pathlib import Path

import pytest

CHECKER = Path(__file__).resolve().parents[2] / "lean/scripts/check-imports.py"
# Python-only CI does not install Lean; the Lean parity job builds this binary first.
pytestmark = pytest.mark.skipif(
    not (CHECKER.parents[1] / ".lake/build/bin/conformance").exists(),
    reason="Lean checker is not built",
)


def check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(root)], capture_output=True, text=True, timeout=30
    )


def tree(root: Path, implementation: str, proofs: str = "import Ssz\n") -> None:
    (root / "Ssz").mkdir()
    (root / "Ssz.lean").write_text(implementation)
    (root / "Ssz/Proofs.lean").write_text(proofs)
    (root / "Ssz/Orphan.lean").write_text("def orphan := 1\n")


@pytest.mark.parametrize(
    "phantom",
    [
        "/-\nimport Ssz.Orphan\n-/\n",
        "/- outer /- nested -/\nimport Ssz.Orphan\n-/\n",
        '-- import Ssz.Orphan\ndef text := "import Ssz.Orphan"\n',
        'def text := "\\"\nimport Ssz.Orphan\n"\n',
    ],
)
def test_phantom_import_does_not_cover_source(tmp_path: Path, phantom: str) -> None:
    tree(tmp_path, phantom)
    result = check(tmp_path)
    assert result.returncode != 0
    assert "missing from the audited import closure:\nSsz.Orphan" in result.stderr


@pytest.mark.parametrize(
    "header",
    [
        "import /- between tokens /- nested -/ -/ Ssz.Orphan\n",
        "module\npublic import Ssz.Orphan\n",
        "module\nmeta import all Ssz.Orphan\n",
    ],
)
def test_real_import_covers_source(tmp_path: Path, header: str) -> None:
    tree(tmp_path, header)
    result = check(tmp_path)
    assert result.returncode == 0, result.stderr


def test_escaped_module_components(tmp_path: Path) -> None:
    tree(tmp_path, "import Ssz.«Odd.name»\nimport Ssz.Orphan\n")
    (tmp_path / "Ssz/Odd.name.lean").write_text("def odd := 1\n")
    result = check(tmp_path)
    assert result.returncode == 0, result.stderr


def test_indirect_proof_import_is_rejected(tmp_path: Path) -> None:
    tree(tmp_path, "import Ssz.Orphan\n")
    (tmp_path / "Ssz/Orphan.lean").write_text("import Ssz.Proofs\n")
    result = check(tmp_path)
    assert result.returncode != 0
    assert "proof modules imported by the implementation:\nSsz.Proofs" in result.stderr


def test_shared_module_cannot_hide_acyclic_proof_dependency(tmp_path: Path) -> None:
    tree(
        tmp_path,
        "import Shared\n",
        "import Ssz\nimport Ssz.Orphan\nimport Ssz.Proofs.Helper\n",
    )
    (tmp_path / "Ssz/Proofs").mkdir()
    (tmp_path / "Ssz/Proofs/Helper.lean").write_text("def helper := 1\n")
    (tmp_path / "Shared.lean").write_text("import Ssz.Proofs.Helper\n")
    result = check(tmp_path)
    assert result.returncode != 0
    assert "proof modules imported by the implementation:\nSsz.Proofs.Helper" in result.stderr
    assert "missing from the audited import closure" not in result.stderr


def test_shared_module_can_cover_owned_implementation(tmp_path: Path) -> None:
    tree(tmp_path, "import Shared\n")
    (tmp_path / "Shared.lean").write_text("import Ssz.Orphan\n")
    (tmp_path / "Unused.lean").write_text("def unused := 1\n")
    result = check(tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("excluded", [".lake", ".git", ".venv"])
def test_dependency_and_build_directories_are_not_parsed(tmp_path: Path, excluded: str) -> None:
    tree(tmp_path, "import Ssz.Orphan\n")
    ignored = tmp_path / excluded / "nested"
    ignored.mkdir(parents=True)
    (ignored / "Ignored.lean").write_text("/- unterminated")
    result = check(tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("header", ["import\n", "/- unterminated", "public import Ssz.Orphan\n"])
def test_invalid_header_is_not_treated_as_complete(tmp_path: Path, header: str) -> None:
    tree(tmp_path, header)
    result = check(tmp_path)
    assert result.returncode != 0
    assert "cannot parse import header" in result.stderr


def native_tree(root: Path, implementation: str) -> None:
    (root / "SszNative").mkdir()
    (root / "SszNative.lean").write_text(implementation)
    (root / "SszNative/Proofs.lean").write_text("import SszNative\n")
    (root / "SszNative/Orphan.lean").write_text("def nativeOrphan := 1\n")


def test_native_phantom_import_does_not_cover_source(tmp_path: Path) -> None:
    tree(tmp_path, "import Ssz.Orphan\n")
    native_tree(tmp_path, "/-\nimport SszNative.Orphan\n-/\n")
    result = check(tmp_path)
    assert result.returncode != 0
    assert "SszNative.Orphan" in result.stderr


def test_native_reference_imports_cover_source(tmp_path: Path) -> None:
    tree(tmp_path, "import Ssz.Orphan\n")
    native_tree(tmp_path, "import Ssz\nimport SszNative.Orphan\n")
    result = check(tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("source", ["Ssz", "SszNative"])
def test_cross_tree_proof_import_is_rejected(tmp_path: Path, source: str) -> None:
    tree(tmp_path, "import Ssz.Orphan\n")
    native_tree(tmp_path, "import SszNative.Orphan\n")
    target = "SszNative" if source == "Ssz" else "Ssz"
    path = tmp_path / f"{source}.lean"
    path.write_text(path.read_text() + f"import {target}.Proofs\n")
    result = check(tmp_path)
    assert result.returncode != 0
    assert "proof modules imported by the implementation" in result.stderr
    assert f"{target}.Proofs" in result.stderr


@pytest.mark.parametrize("source", ["Ssz", "SszNative"])
def test_shared_bridge_cannot_hide_cross_tree_proof(tmp_path: Path, source: str) -> None:
    tree(tmp_path, "import Ssz.Orphan\n")
    native_tree(tmp_path, "import SszNative.Orphan\n")
    target = "SszNative" if source == "Ssz" else "Ssz"
    proof_dir = tmp_path / target / "Proofs"
    proof_dir.mkdir()
    (proof_dir / "Helper.lean").write_text("theorem helper : True := True.intro\n")
    proof_root = tmp_path / target / "Proofs.lean"
    proof_root.write_text(proof_root.read_text() + f"import {target}.Proofs.Helper\n")
    (tmp_path / "Shared.lean").write_text(f"import {target}.Proofs.Helper\n")
    implementation = tmp_path / f"{source}.lean"
    implementation.write_text(implementation.read_text() + "import Shared\n")
    result = check(tmp_path)
    assert result.returncode != 0
    assert "proof modules imported by the implementation" in result.stderr
    assert f"{target}.Proofs.Helper" in result.stderr
