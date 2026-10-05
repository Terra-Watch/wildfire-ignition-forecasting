import pytest

from deep_multimodal.synthetic import write_synthetic_package


@pytest.fixture(scope="session")
def package(tmp_path_factory):
    return write_synthetic_package(tmp_path_factory.mktemp("package"))
