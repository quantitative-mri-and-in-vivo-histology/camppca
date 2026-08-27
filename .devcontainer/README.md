# Devcontainer

A Docker Compose–based devcontainer for camppca, with Claude Code preinstalled.
Two configs share one image definition and one set of lifecycle scripts.

## Picking a config

`Ctrl+Shift+P` → **Dev Containers: Reopen in Container** → pick one:

| Config | Folder | What it adds |
|---|---|---|
| **Dev Container** | `.devcontainer/` | CPU. Plain `jax`. |
| **Dev Container (GPU)** | `.devcontainer/gpu/` | CUDA 12.4 base image, all host GPUs, `jax[cuda12]`. |

The denoising engine is JAX, so prefer the GPU config on a machine that has one.
It **fails** where there is no GPU — deliberately; use the default config there.

## First run

`docker-compose.override.yml` is gitignored but **required** — both configs list
it. If it is missing:

```bash
cp .devcontainer/docker-compose.override.yml.example .devcontainer/docker-compose.override.yml
```

The stub is valid as-is and adds nothing. You only need `.env` if you add a mount
that interpolates a variable — see `.env.example`.

## How the files layer

Dev Containers merges the compose files listed in a config, later files winning:

```
docker-compose.yml                  base: build, volumes, env, shm_size
  + docker-compose.override.yml     your host paths            (gitignored)
      + gpu/docker-compose.gpu.yml  CUDA image + GPU reservation
```

Machine-specific paths belong in `docker-compose.override.yml`, and the paths
themselves in `.env` — so even a mis-committed override file leaks nothing.

Editor settings are **not** in `devcontainer.json`. They live in
`../.vscode/settings.json`, so both configs share one copy and the settings also
apply outside a container. The configs differ only in `name` and
`dockerComposeFile`.

## Python layout

- **Image** (`requirements.txt`) — tooling only: pytest, ruff, mypy, debugpy,
  Jupyter, matplotlib. Baked in so a fresh container can lint and test
  immediately.
- **Project** (`../pyproject.toml`) — camppca's real dependencies. Installed
  editable by `post-create.sh`: `uv pip install --system -e "/workspace[dev]"`.
- **GPU** (`gpu/gpu-requirements.txt`) — `jax[cuda12]`, installed last so it
  overwrites the CPU jaxlib from the project install.

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
| Add a system package | `Dockerfile` **and** `gpu/Dockerfile` |
| Mount a dataset | `docker-compose.override.yml` + `.env` |
| Change timezone | `TZ` in `.env` (defaults to `Europe/Berlin`) |
| Add a VS Code extension | both `devcontainer.json` + `../.vscode/extensions.json` |
| Change the CUDA wheels | `gpu/gpu-requirements.txt` |

The `Dockerfile` / `gpu/Dockerfile` pair and the two `devcontainer.json` files are
the places that must be kept in sync by hand; each carries a comment saying so.
