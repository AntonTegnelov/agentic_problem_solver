# Agentic_problem_solver dev container

An SSH-reachable Linux dev box for Agentic_problem_solver, following the house
pattern in `E:\Programmering\Code\DEVCONTAINER_TEMPLATE.md` — **this one uses
host port 2238** (see the template's port table for the full allocation).

## What's inside

| Tool | Version / notes |
|---|---|
| Python | 3.12 (`python:3.12-slim-bookworm` base); venv at `/home/dev/.venv` on a named volume, first on PATH |
| Node.js | 22 (NodeSource) — only for Claude Code |
| Claude Code | latest, installed globally via npm |
| sshd | hardened: pubkey-only, no root, `dev` user only, host port **2238** |

The repo is bind-mounted at `/workspaces/Agentic_problem_solver`, so edits sync
both ways. The gitignored `.env` (with `GEMINI_API_KEY`) arrives via the mount
— it is never baked into the image (`.dockerignore` keeps it out of the build
context).

**The container venv lives at `/home/dev/.venv`, never the repo's `.venv/`** —
that one is the Windows venv (its launchers are `.exe` files). `which python`
inside the container must print `/home/dev/.venv/bin/python`.

**Named volumes** (survive recreates, including Zed's dev-container flow):
`~/.venv` (created on first start by the entrypoint), `~/.cache` (pip cache),
`~/.claude`, `~/.ssh` (incl. the PAT store), and the sshd host keys.

## One-time setup

Make sure `%USERPROFILE%\.ssh\authorized_keys` on the host contains your
public key:

```
type %USERPROFILE%\.ssh\id_ed25519.pub >> %USERPROFILE%\.ssh\authorized_keys
```

The entrypoint installs this file into the container on every start, so key
changes only need `docker compose restart` — no rebuild.

## Build and start

```
docker compose -f .devcontainer/docker-compose.yml up -d --build
```

The container auto-restarts with Docker Desktop (`restart: unless-stopped`).

Stop: `docker compose -f .devcontainer/docker-compose.yml down`
(add `-v` to also wipe the venv, the pip cache, the Claude login, the PAT
store, and ssh host keys).

## Connect

Add to `~/.ssh/config` on the host:

```
Host aps-dev
    HostName localhost
    Port 2238
    User dev
```

Then `ssh aps-dev`, or point Claude Code / Cursor / JetBrains Gateway at it.
Zed: `zed ssh://dev@localhost:2238/workspaces/Agentic_problem_solver`, or
"Reopen in Dev Container" (recreates once on first attach — state is on
volumes, so it survives). VS Code: "Dev Containers: Reopen in Container",
which runs the editable install via `postCreateCommand`.

## First-login project setup (SSH users)

```bash
cd /workspaces/Agentic_problem_solver
pip install -e . -r requirements.txt   # into /home/dev/.venv (already on PATH)
pytest
ruff check .
APS --help
```

## Git identity / push

Commit identity and the credential helper are baked into the image's system
gitconfig — nothing to configure. Pushing uses a **fine-grained per-repo
PAT** over https, **never an SSH key**: GitHub SSH keys can't be scoped to
one repo, and this container must not reach beyond its own repo (see
DEVCONTAINER_TEMPLATE.md).

One-time, after minting the PAT (GitHub -> Settings -> Developer settings ->
Fine-grained tokens -> Repository access: only
`AntonTegnelov/agentic_problem_solver` -> Permissions: Contents = Read and
write):

```bash
printf 'https://AntonTegnelov:%s@github.com\n' '<the PAT>' > ~/.ssh/git-credentials
chmod 600 ~/.ssh/git-credentials
```

(Or just `git push` once and answer the prompt — username `AntonTegnelov`,
password = the PAT; the credential helper writes the same file.) The store
lives on the `ssh-config` named volume, so the login survives container
recreates and rebuilds.

## Claude Code

`claude` is preinstalled. Log in once (`claude` -> follow the OAuth flow);
credentials live on the `claude-config` named volume and survive rebuilds.

## Troubleshooting

- **`Permission denied (publickey)`** — check `%USERPROFILE%\.ssh\authorized_keys`
  contains your pubkey, then `docker compose -f .devcontainer/docker-compose.yml restart`.
- **Host key changed after `down -v`** — the host-key volume was wiped; run
  `ssh-keygen -R "[localhost]:2238"` on the host and reconnect.
- **`python` is the wrong one** — `which python` should print
  `/home/dev/.venv/bin/python`. If the venv is missing, restart the container;
  the entrypoint recreates it (then re-run the `pip install -e` line).
- **`ModuleNotFoundError: No module named 'toml'` during `pip install -e .`** —
  `setup.py` imports `toml` at build time; `pyproject.toml` lists it under
  `[build-system].requires` so isolated builds get it. If you see this, the
  `requires` line was changed — restore `"toml"` there.
