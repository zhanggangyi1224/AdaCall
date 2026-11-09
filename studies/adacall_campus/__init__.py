"""adacall_campus — AdaCall's group-spillover testbed. Importing the package registers its four
abc implementations (adacall_cd.env / .pop / .decision / .collector) for engine.build_simulator."""

from __future__ import annotations

from . import model  # noqa: F401  (side effect: @register decorators run)
from .bundles import make_bundles, target_dorms  # noqa: F401

__all__ = ["make_bundles", "model", "target_dorms"]
