# TeRL

Terminal-agent RL built on [slime](https://github.com/THUDM/slime).

slime is vendored as a git submodule pinned to **v0.3.1** (`a6272da`).

## Setup

```bash
git clone --recurse-submodules <url> TeRL
cd TeRL
```

Already cloned without submodules?

```bash
git submodule update --init
```

Install slime (it uses `setup.py`; `pyproject.toml` carries only tool config, no
`[project]` table):

```bash
pip install -e ./slime
```

Verify the pin:

```bash
git submodule status
#  a6272da0d4f3d0a08520c99a2f3b4f6c887960dc slime (v0.3.1)
```
