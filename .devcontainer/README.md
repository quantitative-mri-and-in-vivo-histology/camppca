# Devcontainer

A Docker Compose–based devcontainer for camppca, with Claude Code preinstalled. CPU
only: camppca runs on numpy and scipy.

`Ctrl+Shift+P` → **Dev Containers: Reopen in Container**.

## First run

`docker-compose.override.yml` is gitignored but **required** — `devcontainer.json`
lists it. If it is missing:

```bash
cp .devcontainer/docker-compose.override.yml.example .devcontainer/docker-compose.override.yml
```

The stub is valid as-is and adds nothing. You only need `.env` if you add a mount
that interpolates a variable — see `.env.example`.

## How the files layer

Dev Containers merges the compose files listed in `devcontainer.json`, later files
winning:

```
docker-compose.yml                  base: build, volumes, env, shm_size
  + docker-compose.override.yml     your host paths            (gitignored)
```

Machine-specific paths belong in `docker-compose.override.yml`, and the paths
themselves in `.env` — so even a mis-committed override file leaks nothing.

Editor settings are **not** in `devcontainer.json`. They live in
`../.vscode/settings.json`, so they also apply outside a container.

## Python layout

- **Image** (`requirements.txt`) — tooling only: pytest, ruff, mypy, debugpy,
  Jupyter, matplotlib. Baked in so a fresh container can lint and test
  immediately.
- **Project** (`../pyproject.toml`) — camppca's real dependencies. Installed
  editable by `post-create.sh`: `uv pip install --system -e "/workspace[dev]"`.

Don't duplicate project dependencies into `requirements.txt` — that is how the
two lists drift.

`PYTHONPATH=/workspace` matches the flat layout (the package is `/workspace/camppca`),
but the editable install is what makes imports work properly (entry points,
metadata, `pip show`).

## Plots

`matplotlibrc` sets `backend: Agg`, which is **non-interactive**: `savefig` and
inline Jupyter / Interactive-window figures work, but `plt.show()` from a plain
script is a no-op. There is no X display in the container. That is intended for
headless runs; use a notebook for interactive plotting.

## Customizing

| Want to… | Edit |
|---|---|
| Add a Python tool for every container | `requirements.txt` |
| Add a project dependency | `../pyproject.toml` |
| Add a system package | `Dockerfile` |
| Mount a dataset | `docker-compose.override.yml` + `.env` |
| Change timezone | `TZ` in `.env` (defaults to `Europe/Berlin`) |
| Add a VS Code extension | `devcontainer.json` + `../.vscode/extensions.json` |
