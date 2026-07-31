"""Repro: change_env restores absent keys as empty strings.

C++ get_bool_env_var rejects empty values and aborts. A later region that
reads TMS_INIT_ENABLE / TMS_INIT_ENABLE_CPU_BACKUP can then crash if a prior
test used change_env on an unset key.
"""

import os
import sys

from torch_memory_saver.utils import change_env


def run(hook_mode: str = "torch"):
    del hook_mode
    key = "TMS_INIT_ENABLE_CPU_BACKUP"
    os.environ.pop(key, None)
    assert key not in os.environ
    with change_env(key, "1"):
        assert os.environ[key] == "1"
    # Bug: key remains present as "" instead of being unset.
    print(f"after_change_env present={key in os.environ!r} value={os.environ.get(key)!r}")
    assert key not in os.environ, (
        f"expected {key} unset after change_env; got value={os.environ.get(key)!r}"
    )


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "torch")
