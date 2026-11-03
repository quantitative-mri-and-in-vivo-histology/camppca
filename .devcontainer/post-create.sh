#!/usr/bin/env bash
# Runs once, after the container is created (devcontainer.json postCreateCommand).
# Must stay idempotent: re-creating a container re-runs it.
set -uo pipefail

# --- SSH ------------------------------------------------------------------
# The host's ~/.ssh is bind-mounted read-mostly at /home/node/.ssh-host. Copy it
# to a real directory so we can fix permissions (ssh refuses group/world-readable
# private keys) without touching the host's files.
if [ -d /home/node/.ssh-host ]; then
  mkdir -p /home/node/.ssh
  cp -r /home/node/.ssh-host/. /home/node/.ssh/ 2>/dev/null || true
  chmod 700 /home/node/.ssh
  chmod 600 /home/node/.ssh/id_* 2>/dev/null || true
  chmod 644 /home/node/.ssh/*.pub 2>/dev/null || true
  echo "post-create: ssh keys staged from host"
fi

# --- Project install ------------------------------------------------------
# Editable install so `import <package>` works everywhere without PYTHONPATH
# games, and so console_scripts entry points get registered. Skipped while the
# package directory is missing.
if [ -f /workspace/pyproject.toml ] && [ -f /workspace/camppca/__init__.py ]; then
  echo "post-create: installing project (editable, with [dev] extras) ..."
  if ! sudo uv pip install --system -e "/workspace[dev]"; then
    echo "post-create: WARNING - editable install failed."
    echo "             Fix pyproject.toml, then re-run:"
    echo "             sudo uv pip install --system -e '/workspace[dev]'"
  fi
else
  echo "post-create: no camppca/__init__.py - skipping editable install."
  echo "             Once the package directory exists, run:"
  echo "             sudo uv pip install --system -e '/workspace[dev]'"
fi

# --- Hand build artifacts back to the container user ----------------------
# The install above runs under sudo, and setuptools writes <pkg>.egg-info (and
# sometimes build/) into the BIND-MOUNTED workspace. Those land root-owned, so
# on the host they cannot be deleted, and a host-side `pip install -e .` fails
# with "error: Cannot update time stamp of directory '<pkg>.egg-info'".
# The node user's UID matches the host user's, so chowning here fixes both sides.
sudo chown -R node:node \
  /workspace/*.egg-info \
  /workspace/src/*.egg-info \
  /workspace/build \
  /workspace/.eggs \
  2>/dev/null || true
