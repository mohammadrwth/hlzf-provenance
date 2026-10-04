from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hlzf.config import PROJECT_ROOT, load_settings
from hlzf.pipeline import open_runtime, run_corpus


def make_settings(data_dir: Path, root: Path = PROJECT_ROOT, **kw):
    """Settings for tests: offline, no key, isolated data dir (override via kwargs)."""
    values = {"offline": True, "api_key": None, "reviewer": "tester", **kw}
    return load_settings(root=root, data_dir=data_dir, **values)


@pytest.fixture(scope="session")
def processed(tmp_path_factory) -> Path:
    """Run the full pipeline once over the synthetic corpus; returns the data dir."""
    data = tmp_path_factory.mktemp("data")
    rt = open_runtime(make_settings(data), log=lambda m: None)
    results = run_corpus(rt, include_real=False)
    assert all(r.status not in ("not extracted", "budget") for r in results), results
    rt.conn.close()
    return data


@pytest.fixture()
def rt(processed, tmp_path):
    """A runtime on a private copy of the processed database (safe to mutate)."""
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(processed / "hlzf.db", data / "hlzf.db")
    for sub in ("synthetic", "cache"):
        if (processed / sub).exists():
            shutil.copytree(processed / sub, data / sub)
    runtime = open_runtime(make_settings(data), log=lambda m: None)
    yield runtime
    runtime.conn.close()


@pytest.fixture()
def conn(rt):
    return rt.conn
