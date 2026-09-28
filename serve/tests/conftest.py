"""Fixtures for the Deem serving-lane tests."""

import pytest
from serve_helpers import live_server


@pytest.fixture
def base_url():
    with live_server() as url:
        yield url
