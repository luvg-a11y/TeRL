---
name: terl-eval
description: Run or debug TeRL checkpoint evaluations on terminal-agent benchmarks (Terminal-Bench Lite, Terminal-Bench 2.0) using Harbor + SGLang. Use when asked to evaluate a checkpoint or model, run a benchmark sweep, interpret eval results, or diagnose a failing Harbor run. Covers the serve/evaluate split, the sandbox-backend constraint, and the failure modes that silently produce wrong numbers.
---

# TeRL evaluation

Evaluating checkpoints with [Harbor](https://github.com/laude-institute/harbor), following
the protocol in *Tmax* ([arXiv:2606.23321](https://arxiv.org/abs/2606.23321)).

Files live in `evaluation/`: `README.md` (runbook), `NOTES.md` (rationale and paper
fidelity), `run_eval.sh` (sweep), `summarize.py` (results table), `Dockerfile`.

## The mental model

Two processes, and conflating them causes most confusion:

- **SGLang owns the model.** It loads weights and serves an OpenAI-compatible endpoint.
- **Harbor owns the evaluation.** Its `-m` flag is a *URL routing string*, never a
  checkpoint path. Harbor never loads weights and never starts a server.

`-m openai/<name>` means "speak the OpenAI protocol", not "call OpenAI". The endpoint comes
from `OPENAI_BASE_URL`. No request leaves the host.

If no server is listening, the run fails. Always start SGLang first and verify it.

## Before running anything

1. **Is a container runtime reachable?** `docker info`. Harbor's default `-e docker`
   backend needs one.
   - **A Runpod pod never has one** and cannot get one: it is an unprivileged container
     without `CAP_SYS_ADMIN` (`grep CapEff /proc/self/status` — anything short of
     `0000003fffffffff`). Docker and rootless Podman both fail. Use `-e daytona`.
   - Do not try to install Docker or Podman on a pod. It is a host policy.
2. **Is the job directory visible to the daemon?** Harbor bind-mounts it into task
   containers, and the daemon resolves that path on *its own* host.
   - Inside slime's image: `/workspace` must be mounted at the same path on both sides.
   - On macOS/colima: run from under `/Users/<you>`; only that is mounted into the VM.
   - Wrong path => every trial fails with `RewardFileNotFoundError` and an empty
     `verifier/`. It looks like a Harbor bug. It is not.
3. **Does SGLang match slime?** `v0.5.15.post1`, CUDA 12.9 (`slime/build_conda.sh:30`).
   slime patches SGLang internals against that exact version. Prefer
   `evaluation/Dockerfile`, which layers Harbor onto `slimerl/slime`.

## Running

Smoke test first — it is fast and catches every wiring problem:

```bash
cd evaluation && ./run_eval.sh --smoke
```

Then the sweep (`--list` prints the plan without touching anything):

```bash
./run_eval.sh
python3 summarize.py jobs
```

Runs are resumable; jobs with an existing `result.json` are skipped. Filter with
`MODELS=` / `DATASETS=`. Never add `--timeout-multiplier` — the paper does not override
timeouts, and changing them makes results incomparable.

For a single model, serve then evaluate:

```bash
python -m sglang.launch_server --model-path <hf-id> --served-model-name m \
  --host 127.0.0.1 --port 8000 --context-length 65536 --mem-fraction-static 0.90
curl -s localhost:8000/v1/models          # verify BEFORE evaluating
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=dummy MSWEA_API_KEY=dummy
harbor run -d openthoughts-tblite@2.0 -a mini-swe-agent -m openai/m -k 5 -n 8 -o /workspace/jobs
```

Use `-a mini-swe-agent`, not `terminus-2`. The paper found Terminus-2 brittle with small
models because it requires sending raw keystrokes.

`MSWEA_API_KEY` is separate from `OPENAI_API_KEY` — mini-swe-agent reads its key there.

## Reading results

**Check `n_errors` before believing any number.** Errored trials are excluded from the
mean, not scored zero, so an infrastructure failure still reports a plausible-looking
score. `summarize.py` flags this; do not skip it.

`n_trials` counts scored trials. A run with many errors is invalid, not merely worse.

Roles: **TB Lite is the selection set, TB 2.0 is the reporting set.** Selecting a
checkpoint on TB 2.0 means selecting on the test set.

Reference point: untrained Qwen3.5-9B scores 41.9 on TB Lite, 16.1 on TB 2.1. Landing far
from that suggests a setup problem rather than a model result.

## Failure modes

| Symptom | Cause |
| --- | --- |
| `Docker is not installed or not on PATH` | no daemon; use `-e daytona` |
| `RewardFileNotFoundError`, empty `verifier/` | job dir not visible to the daemon |
| `cannot clone: Operation not permitted` | Podman; no `CAP_SYS_ADMIN`. Not fixable on a pod |
| Many timeouts | SGLang too slow — lower `-n`. Throughput costs score |
| Empty agent trajectories | thinking model consuming the budget; `--ak reasoning_effort=none` |
| OOM on 27B / 35B-A3B | raise `--tp-size`, lower `-n`, or `--quantization fp8` |

## Comparability

Hold the serving setup fixed across everything you compare. Per-task timeouts convert
inference throughput into score, so two checkpoints served on different hardware are not
comparable. The paper pinned every model to a single A100 for exactly this reason.

Harbor's `mini-swe-agent` exposes no temperature control, so each model falls back to its
own `generation_config.json`. Pin it server-side with `--override-generation-config` if
models in one sweep must sample identically. `NOTES.md` records which settings come from
the paper and which are ours.
