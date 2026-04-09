"""
Shared environment instances used by both the HTTP API and the Gradio UI.

Keeping them here as module-level singletons guarantees that an action
executed through the browser affects the same environment that the
/step, /state, and /score endpoints read.
"""
from __future__ import annotations

import sys
import os

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT  = os.path.dirname(_SERVER_DIR)
for _p in (_SERVER_DIR, _PROJ_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env import IncidentResponseEnv

env           = IncidentResponseEnv()
_baseline_env = IncidentResponseEnv()
