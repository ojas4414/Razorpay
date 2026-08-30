"""Test configuration.

The engine loads vocab.json and policy_rl.onnx by relative path, so tests run
from the repository root regardless of where pytest was invoked.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session", autouse=True)
def _run_from_repo_root():
    previous = os.getcwd()
    os.chdir(ROOT)
    yield
    os.chdir(previous)
