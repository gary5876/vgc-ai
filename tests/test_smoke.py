"""Smoke test — verifies the package imports and the test harness works.

Replace or extend once real code exists.
"""

import vgc_ai


def test_package_has_version() -> None:
    assert isinstance(vgc_ai.__version__, str)
    assert vgc_ai.__version__
