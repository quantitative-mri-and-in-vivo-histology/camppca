"""Runs inside a pool worker. Imports nothing, so it cannot load JAX itself."""

import sys


def jax_loaded():
    return "jax" in sys.modules
