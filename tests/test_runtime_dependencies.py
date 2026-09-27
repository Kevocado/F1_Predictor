"""Every third-party module imported under src/ must be a runtime dependency.

This caught a production outage. `api/explain.py` imports `httpx` at runtime,
but F1 did not declare it at all and NBA listed it only under the `dev` extra,
so `docker build && docker run` produced an image that died on import with
`ModuleNotFoundError: No module named 'httpx'`. Both services were rolled back
from the VPS.

The test suite could not see it: httpx arrives in the venv through dev/test
tooling (fastapi's own test client, respx, pytest tooling), so every test passed
and the image was broken. That is the whole point of this test — it reads
pyproject.toml, not the venv, so it asks the question the venv cannot answer.

A distribution can satisfy more than one module name, so there is a small
explicit map. Keep it short and keep it honest: a new import with no entry here
fails the test, which is the intended pressure.
"""
from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

# Top-level module name -> import name on PyPI, where they differ.
MODULE_TO_DISTRIBUTION = {
    "attr": "attrs",
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "dotenv": "python-dotenv",
    "jwt": "pyjwt",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    # Vendored in-tree, or a stdlib module we must not look for.
    "pl_predictor": None,
    "nba_predictor": None,
    "nfl_predictor": None,
    "cfb_predictor": None,
    "f1_predictor": None,
    "fastf1": "fastf1",
}

STDLIB = set(getattr(__import__("sys"), "stdlib_module_names", ()))


def _declared() -> set[str]:
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    runtime = data["project"]["dependencies"]
    return {re.split(r"[<>=!\[ ]", dep.strip())[0].lower().replace("_", "-") for dep in runtime}


def _imported_modules() -> set[str]:
    """Every top-level module imported by first-party code under src/."""
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # a relative import: in-tree by definition
                    continue
                if node.module:
                    found.add(node.module.split(".")[0])
    return found


def test_the_sweep_still_sees_a_real_runtime_import():
    """Guards the sweep below against passing because it found nothing.

    The canary used to be `httpx`, on the grounds that `api/explain.py` imported
    it in every one of these repos. That is no longer true here: this is the
    update its own docstring asked for. The proxy was F1's only httpx user, and
    with it gone the sweep had nothing left to check — so `httpx` as a canary
    would now be asserting a fact about the code that had stopped being true,
    which is a test that fails for a reason nobody can act on.

    `fastapi` and `pydantic` are canaries that are true, and true for a boring
    reason: both are imported at runtime and either would break the image on
    import if it were ever dropped. A canary has to be a fact that cannot
    quietly stop being a fact.
    """
    found = _imported_modules()
    for canary in ("fastapi", "pydantic"):
        assert canary in found, (
            f"the sweep no longer sees {canary}, so the sweep below is passing "
            f"because it found nothing: update this test"
        )


def test_httpx_is_no_longer_a_runtime_import():
    """The proxy was its only user, and that is now worth keeping true.

    Not because an unused dependency is dangerous — it is not — but because
    pyproject.toml still lists `httpx` as a runtime dependency *because
    api/explain.py imports it*, and that reason is gone. The dependency stays (the
    tests need it, and dropping a declared runtime dep is a separate change with
    an image-build risk that cannot be checked from here), but if a future change
    starts importing httpx at runtime again this fails, and the pyproject comment
    can be made true again instead of staying a fossil.
    """
    assert "httpx" not in _imported_modules(), (
        "httpx is imported at runtime again. Either pyproject.toml's stated "
        "reason for listing it as a runtime dependency needs restoring, or it is "
        "now test-only and should move to the dev extra."
    )


@pytest.mark.parametrize("module", sorted(_imported_modules()))
def test_import_is_a_runtime_dependency(module: str):
    if module in STDLIB:
        pytest.skip(f"{module} is stdlib")
    if module.startswith("_"):
        pytest.skip("private module")

    if module in MODULE_TO_DISTRIBUTION:
        distribution = MODULE_TO_DISTRIBUTION[module]
        if distribution is None:
            pytest.skip(f"{module} is first-party")
    else:
        distribution = module.lower().replace("_", "-")

    declared = _declared()
    assert distribution in declared or module.lower().replace("_", "-") in declared, (
        f"{module} is imported under src/ but {distribution!r} is not in "
        f"[project].dependencies. A Docker image installs only the runtime "
        f"dependencies, so this crashes at import on the VPS while every test "
        f"passes (test tooling pulls it in). Add it there; a dev extra will not do."
    )
