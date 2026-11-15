import pytest

from amu_mem0 import GatedMemory
from amu_mem0.testing import offline_memory

SCOPE = {"agent_id": "shared-analyst"}


@pytest.fixture
def memory(tmp_path):
    m = offline_memory(str(tmp_path))
    yield m
    m.close()


@pytest.fixture
def gm(memory):
    return GatedMemory(memory)


def add_id(gm, text, **kw):
    """Add through the gate and return the new memory id."""
    kw.setdefault("agent_id", SCOPE["agent_id"])
    res = gm.add(text, **kw)
    return res["results"][0]["id"]


def ids(result):
    return [r["id"] for r in result["results"]]
