import os

import pytest

os.environ.setdefault("EB_EMBEDDING_BACKEND", "tfidf")  # tests must not download models

from src.pipeline import build_system  # noqa: E402


@pytest.fixture(scope="session")
def system(tmp_path_factory):
    root = tmp_path_factory.mktemp("eb")
    return build_system(data_dir=root / "data", artifact_dir=root / "artifacts", regenerate=True,
                        embedding_backend="tfidf")
