"""Runnable, offline version of the README quickstart.

In your own code, replace ``offline_memory(...)`` with your existing
``mem0.Memory(...)`` instance; nothing else changes.
"""

import tempfile

from amu_mem0 import GatedMemory
from amu_mem0.testing import offline_memory

memory = GatedMemory(offline_memory(tempfile.mkdtemp()))

payroll = memory.add(
    "Q3 payroll total is 4.2M",
    agent_id="analyst",
    sources={"hris": ["hr.salary"]},
)["results"][0]["id"]
memory.add("Payroll cost grew 8% QoQ", agent_id="analyst", derived_from=[payroll])

marketing = memory.search("payroll", filters={"agent_id": "analyst"}, entitlements={"mkt.campaign"})
finance = memory.search("payroll", filters={"agent_id": "analyst"}, entitlements={"hr.salary"})

print("marketing sees:", [r["memory"] for r in marketing["results"]])
print("marketing denied:", [(d["reason"], d["unentitled_labels"]) for d in marketing["amu_denied"]])
print("finance sees:", sorted(r["memory"] for r in finance["results"]))
memory.memory.close()
