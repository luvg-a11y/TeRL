# Evaluation

Evaluate checkpoints on Terminal-Bench Lite and Terminal-Bench 2.0 with Harbor.

Requires a GPU host. Background and design rationale: [NOTES.md](./NOTES.md).
No GPU: [LOCAL_TESTING.md](./LOCAL_TESTING.md).

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv tool install harbor
pip install "sglang[all]"
```

## Single run

```bash
# serve (leave running)
export HF_HOME=/workspace/hf
python -m sglang.launch_server --model-path Qwen/Qwen3.5-9B \
  --served-model-name m --host 127.0.0.1 --port 8000 \
  --context-length 65536 --mem-fraction-static 0.90

# evaluate
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy
export MSWEA_API_KEY=dummy

harbor run -d openthoughts-tblite@2.0 -a mini-swe-agent -m openai/m -k 5 -n 8
```

Add `-e daytona` (with `DAYTONA_API_KEY`) if the host has no Docker daemon — a Runpod pod
never does.

## Sweep

```bash
./run_eval.sh --list      # print plan
./run_eval.sh --smoke     # 1 task, 1 attempt, 1 repeat
./run_eval.sh             # 5 models x 2 benchmarks x 3 repeats
python3 summarize.py jobs # results table
```

Resumable: jobs with an existing `result.json` are skipped.

Filter: `MODELS="qwen3.5-9b qwen3.6-27b" DATASETS=tblite ./run_eval.sh`

Override: `K` `REPEATS` `N_CONCURRENT` `LIMIT` `HARBOR_ENV` `AGENT_KWARGS` `CONTEXT_LENGTH` `MEM_FRACTION`

## Models

| Key | HF id | Params | bf16 | TP |
| --- | --- | --- | --- | --- |
| `qwen3.5-2b` | `Qwen/Qwen3.5-2B` | 2.3B | ~5 GB | 1 |
| `qwen3.5-4b` | `Qwen/Qwen3.5-4B` | 4.7B | ~9 GB | 1 |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | 9.7B | ~19 GB | 1 |
| `qwen3.6-27b` | `Qwen/Qwen3.6-27B` | 27.8B | ~56 GB | 1 |
| `qwen3.6-35b-a3b` | `Qwen/Qwen3.6-35B-A3B` | 36.0B | ~72 GB | 2 |

`qwen3.6-35b-a3b` does not fit one 80 GB card; use TP 2 or `--quantization fp8`.
Models needing more GPUs than the host has are skipped.

## Datasets

| Key | Dataset | Tasks |
| --- | --- | --- |
| `tblite` | `openthoughts-tblite@2.0` | 100 |
| `tb2` | `terminal-bench@2.0` | 89 |

Also available: `hello-world@1.0` (1), `terminal-bench-sample@2.0` (10).

## Results

```bash
python3 summarize.py jobs   # mean +/- stderr across repeats
harbor view jobs            # trajectory browser
```

Per-run artifacts: `jobs/<model>__<dataset>__rep<n>/` — `agent/` trajectory,
`verifier/reward.txt`, `exception.txt` on failure.

Check `n_errors` in `result.json`. Errored trials are excluded from the mean, not scored
zero, so a broken run still reports a plausible number.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Docker is not installed or not on PATH` | `-e daytona`; a Runpod pod cannot run Docker |
| `RewardFileNotFoundError`, empty `verifier/` | job dir not visible to the runtime; keep `-o` under `/workspace` |
| Many timeouts | SGLang too slow: lower `N_CONCURRENT`, check `nvidia-smi` |
| OOM on 27B / 35B | raise TP, lower `N_CONCURRENT`, or `--quantization fp8` |
| Empty agent trajectories | `AGENT_KWARGS=reasoning_effort=none` |
| Torch/CUDA mismatch on pod boot | redeploy with CUDA pinned in the filters |
