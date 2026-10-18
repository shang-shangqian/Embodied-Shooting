"""Compatibility entry point; implementation: test11.speed_prediction.taq_gru.taq_velocity_training."""

from pathlib import Path
import sys

package_parent = str(Path(__file__).resolve().parents[2])
if package_parent not in sys.path:
    sys.path.insert(0, package_parent)

from test11.speed_prediction._compat import forward_module

forward_module(__name__, "taq_gru.taq_velocity_training", globals())
