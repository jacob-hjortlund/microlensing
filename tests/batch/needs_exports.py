"""The skip condition of the tests that need the IPM libraries' batch exports."""
import unittest

from microlensing.IPM import lib_ipm, lib_ipm_double

HAS_EXPORTS = hasattr(lib_ipm.lib, "set_worker") and hasattr(lib_ipm_double.lib, "set_worker")
NO_EXPORTS = "the IPM library lacks the batch exports; rebuild it with `source compile` (compile.bat on Windows)"
NEEDS_EXPORTS = unittest.skipUnless(HAS_EXPORTS, NO_EXPORTS)
