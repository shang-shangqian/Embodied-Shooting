"""Compatibility entry point; implementation: speed_prediction.data_tools.audit_stage1_ideal_v2."""

from pathlib import Path
import sys

package_parent = str(Path(__file__).resolve().parent.parent)
if package_parent not in sys.path:
    sys.path.insert(0, package_parent)

from speed_prediction._compat import forward_module

forward_module(__name__, "data_tools.audit_stage1_ideal_v2", globals())
