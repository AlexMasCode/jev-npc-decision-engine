# NPC Decision Engine powered by local Jev / System One

A small terminal application that asks a local model how a game NPC should act and how dangerous the encounter is. Python validates the scenario, submits two typed questions, checks the response and displays the result. There is no action fallback, random policy or Python personality-to-action rule.

This project uses **[amithgc/local-jev](https://github.com/amithgc/local-jev)**, an independent open-source Jev-compatible implementation, not TypeSafe's hosted Jev or its proprietary model. Its LLM backend reads probabilities for fixed answer tokens rather than generating a reply. The application uses its real HTTP API; mocks exist only in tests.

```text
Scenario JSON
     |
     v
Python application (validation + HTTP)
     |
     v
Local Jev / System One
     |
     +---- Choice ----> NPC action
     |
     +---- Score -----> danger
     |
     v
Deterministic Python game logic (panic when danger >= 0.9)
     |
     v
Terminal output
```

## Verified interface

The upstream README and source were inspected on October 6, 2026 at revision
`64a0b31ff343dca32142496cf9edca5a0174a18d` (package version 0.2.0):

- [Installation and API documentation](https://github.com/amithgc/local-jev/blob/64a0b31ff343dca32142496cf9edca5a0174a18d/README.md)
- [Request/response schema](https://github.com/amithgc/local-jev/blob/64a0b31ff343dca32142496cf9edca5a0174a18d/src/local_jev/schema.py)
- [Choice and Score response construction](https://github.com/amithgc/local-jev/blob/64a0b31ff343dca32142496cf9edca5a0174a18d/src/local_jev/primitives.py)
- [Dependencies and command entry point](https://github.com/amithgc/local-jev/blob/64a0b31ff343dca32142496cf9edca5a0174a18d/pyproject.toml)

The client posts to `http://127.0.0.1:8765/v1/systemone` with `state`, `model` and named `questions`. It reads `model` and `answers.action` / `answers.danger`. The questions are evaluated independently; the danger question does not see the action answer.

**Choice** supplies a map of action names to semantic descriptions. The response contains the selected `choice`, option `probabilities` and `confidence`. The four options are `attack`, `flee`, `negotiate` and `call_guards`; descriptions explain combat, escape, peaceful resolution and seeking nearby assistance.

**Score** supplies an ordered list of level descriptions. The result is the probability-weighted average of level indices, not an arbitrary numeric min/max. Our two levels, completely safe and extremely dangerous, have indices 0 and 1. Consequently the danger score is already in [0, 1]. It is a rubric estimate, not a measured probability of injury.

The client prints probabilities and confidence only if returned. Upstream confidence describes how concentrated an option distribution is; it is not a guarantee of correctness. Missing optional confidence is never invented. Invalid actions, nonfinite values, incompatible rubrics and malformed distributions are rejected.

## Prerequisites and WSL2

Use Windows 11 with an **Ubuntu WSL2 distribution**, uv-managed Python 3.12, Git, curl, internet access for installation/model download, and sufficient disk/RAM/VRAM. Run the project and server from the same Ubuntu distribution. No Docker, database or frontend is required by this application. The upstream server owns its own dependencies, including its HTTP server framework.

The CUDA model also needs a C compiler for Triton's runtime kernels. If `command -v cc` returns nothing, install the standard Ubuntu build tools:

```bash
sudo apt update
sudo apt install -y build-essential
```

This is a compiler toolchain, not an NVIDIA display driver. The simple CUDA tensor check can pass while model warm-up fails if the compiler is missing.

If Ubuntu is missing, the Windows setup step is manual, outside this project's installation scope. In an elevated PowerShell run:

```powershell
wsl --install -d Ubuntu
wsl --list --verbose
```

Complete the Ubuntu first-run user setup and any requested restart. Do not use the internal `docker-desktop` distribution as the development environment.

Open Ubuntu, then inspect it **before installing model dependencies**:

```bash
cd /mnt/d/IT/_Code/Python/NLP/Jev
python3 --version
command -v python3
command -v pip
nvidia-smi
```

The GPU should appear inside WSL, not just Windows. The NVIDIA driver comes from the Windows host; do not install a Linux display driver in WSL. Ubuntu's system Python is independent of the project's managed interpreter. On this machine it is 3.14.4; the project uses **Python 3.12.15** in both environments.

Install uv in the Ubuntu user's environment with the [official installer](https://docs.astral.sh/uv/getting-started/installation/). The installer was downloaded and inspected before execution. No sudo or shell-profile modification was needed:

```bash
curl -fLsS https://astral.sh/uv/install.sh -o /tmp/jev-uv-install.sh
less /tmp/jev-uv-install.sh
UV_NO_MODIFY_PATH=1 sh /tmp/jev-uv-install.sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
```

uv 0.12.23 and managed Python 3.12.15 were installed for the live verification. This does not replace Ubuntu's system Python.

## Install the application

The client needs only `requests`; tests use standard-library `unittest`.

```bash
cd /mnt/d/IT/_Code/Python/NLP/Jev
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

If `.venv` was created on Windows, it cannot be reused in Linux. It has already been preserved as `.venv-windows` on this machine. Do not rename or recreate the now-working Linux environments during an ordinary run.

**Mounted-drive workaround used here:** this machine's `/mnt/d` mount rejects some package copy/permission operations with `Operation not permitted`. Both Linux environments were therefore created on Ubuntu's native filesystem and linked into the project:

```bash
# Use these instead of the two client setup commands above on an equivalent fresh setup.
# The .venv and .venv-server names must be absent before creating the links.
uv venv --python 3.12 "$HOME/.local/share/jev-runtime/client"
uv venv --python 3.12 "$HOME/.local/share/jev-runtime/server"
ln -s "$HOME/.local/share/jev-runtime/client" .venv
ln -s "$HOME/.local/share/jev-runtime/server" .venv-server
uv pip install --python .venv/bin/python -r requirements.txt
```

The current targets are `/home/alex/.local/share/jev-runtime/client` and `server`. Earlier incomplete mounted-drive attempts are preserved and ignored as `.venv-incomplete*` and `.venv-server-incomplete`. All run commands still use `.venv/bin/python` and `.venv-server/bin/python`.

## Install the real local-jev server

The upstream repository was cloned into `vendor/local-jev` during implementation. If obtaining a fresh project copy, clone it as follows. Source and model caches stay in `Jev`; uv's runtime environments and build cache use the Ubuntu user filesystem.

```bash
cd /mnt/d/IT/_Code/Python/NLP/Jev
# Only needed when vendor/local-jev is absent:
git clone https://github.com/amithgc/local-jev.git vendor/local-jev

git -C vendor/local-jev checkout 64a0b31ff343dca32142496cf9edca5a0174a18d
# Skip environment creation if using the linked environment created above.
uv venv --python 3.12 .venv-server
uv pip install --python .venv-server/bin/python -e ./vendor/local-jev
```

This is the upstream documented uv editable installation, using a separate environment for the heavy server packages. It downloads PyTorch, transformers and the upstream dependencies. The source revision is pinned for API reproducibility; its transitive dependencies are not locked. Keep TLS verification enabled when installing packages.

On this machine, editable and regular builds on `/mnt/d` both failed while writing setuptools metadata. The exact pinned Git archive was extracted to Ubuntu's filesystem and installed as a regular package, without modifying upstream source:

```bash
mkdir -p "$HOME/.local/share/jev-runtime/local-jev"
git -c safe.directory="$PWD/vendor/local-jev" -C vendor/local-jev \
  archive 64a0b31ff343dca32142496cf9edca5a0174a18d | \
  tar -x -C "$HOME/.local/share/jev-runtime/local-jev"
uv pip install --python .venv-server/bin/python "$HOME/.local/share/jev-runtime/local-jev"
```

Git's mounted-drive differences were checked with `core.fileMode=false` and `--ignore-space-at-eol`; there were no source content changes. The installed package is `local-jev==0.2.0`, with PyTorch `2.14.1+cu130` and transformers `5.18.0`. CUDA worked with this initial PyTorch build, so no wheel replacement or display-driver installation was needed. The captured dependency versions are in `results/live/server-requirements.txt`.

Verify the **server environment's** CUDA support and actual kernel execution:

```bash
.venv-server/bin/python scripts/verify_cuda.py
```

If `nvidia-smi` works but the tensor check fails, inspect the installed PyTorch CUDA build and GPU support. Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose a current compatible CUDA wheel rather than guessing a wheel version. Do not infer model CUDA support solely from the CUDA version shown by `nvidia-smi`.

## Start Jev

In the first Ubuntu terminal:

```bash
cd /mnt/d/IT/_Code/Python/NLP/Jev
export HF_HOME="$PWD/.cache/huggingface"
.venv-server/bin/local-jev serve --model llm-qwen3.5-4b
```

For a background process with a recorded PID and log, use `bash scripts/start_server.sh` instead. The helper refuses to launch a duplicate recorded process. Startup messages go to `results/live/server.log`, and the PID is in `results/live/server.pid`.

This explicitly selects the upstream Qwen model for the semantic personality experiment. The first start downloads its weights; allow time and storage for that download. Wait until model loading/warm-up succeeds and the server reports the API address. CUDA is selected automatically when PyTorch can use it. GPU memory availability, including other processes, can prevent loading; a detected GPU does not guarantee this model fits.

If resources are limited, start the smaller upstream model instead:

```bash
.venv-server/bin/local-jev serve --model llm-qwen3.5-2b
```

For the smallest download and a classification-oriented baseline:

```bash
.venv-server/bin/local-jev serve --model nli-deberta-large
```

Use one server command at a time. The smaller models may be less sensitive to personality context. Startup without an explicit model can select a cached model or the small NLI model, so the main experiment specifies Qwen explicitly. No API key is required for the default local server. This educational client targets that unauthenticated local setup.

Check readiness from a second Ubuntu terminal:

```bash
curl --fail http://127.0.0.1:8765/healthz
curl --fail http://127.0.0.1:8765/v1/models
```

## Run

```bash
cd /mnt/d/IT/_Code/Python/NLP/Jev
.venv/bin/python main.py scenarios/wounded_npc.json
.venv/bin/python main.py --all
```

Capture raw responses and a controlled personality comparison from the real running server:

```bash
.venv/bin/python scripts/verify_live.py
```

This checks `/healthz` and `/v1/models`, sends a direct Choice/Score request, records all six scenario responses, and runs the three personalities again with the same display name. Captures go to ignored `results/live/`; they contain real requests, responses and elapsed times, with no model weights or secrets.

The equivalent command after `source .venv/bin/activate` is `python main.py scenarios/wounded_npc.json`. `--all` locates bundled scenarios relative to the application, even when launched from another working directory. All scenario files are validated before the first request. Requests run sequentially, and a batch stops on the first failure with exit code 1. Argument errors use exit code 2; interruption uses 130.

Optional configuration:

```bash
export JEV_BASE_URL=http://127.0.0.1:8765
export JEV_MODEL=jev-latest
.venv/bin/python main.py --all --timeout 180
```

`jev-latest` means the server's selected default. `JEV_MODEL` can select another upstream model; it may trigger an additional download. Supply only the server base URL, without `/v1/systemone`. Timeout defaults to 120 seconds. The application performs no automatic retries or substitute decisions.

## Scenarios and personality experiment

| File | Design | Intended tendency (not a required answer) |
|---|---|---|
| `friendly_player.json` | Trusted unarmed player, healthy neutral NPC | negotiate |
| `wounded_npc.json` | Wounded coward, armed hostile player, no guards | flee |
| `guards_nearby.json` | Armed hostile player, vulnerable neutral NPC, guards available | call_guards |
| `aggressive_npc.json` | Moderate tension, aggressive personality | attack |
| `coward_npc.json` | Same world state, coward personality | flee |
| `diplomatic_npc.json` | Same world state, diplomatic personality | negotiate |

The final three files have exactly equal health, reputation, weapon, guards and distance fields. Only scenario name and personality differ. The personality and its description are sent as context to Jev. Inspect the chosen action and full distributions to see whether personality affected the result; the model may select the same action more than once. These are an experiment, not preassigned outputs.

Health fields must be integers 0-100, reputation an integer -100 to 100, distance a finite number >= 0 in meters, flags actual JSON booleans, and personality one of the four supported names. Unknown/missing fields, malformed JSON, empty names and unreadable files fail cleanly.

## Example terminal output

**Actual output from the live WSL/CUDA run on October 6, 2026:**

```text
NPC Decision Engine
===================

Scenario: Wounded NPC
NPC personality: coward
NPC health: 20
Player health: 90
Player reputation: -60
Player weapon drawn: yes
Guards nearby: no
Distance: 2 meters

Jev decision
------------
Model: llm-qwen3.5-4b
Action: FLEE
Danger score: 0.8719
Choice confidence: 0.9247
Score confidence: 0.7438

Action probabilities:
attack       0.0372
flee         0.9435
negotiate    0.0134
call_guards  0.0059

Application state:
NORMAL MODE
```

Python only derives panic mode from the returned danger threshold. It does not choose or override actions. If the model chooses an impractical action, such as calling absent guards, the demonstration reports that result so the model limitation is visible.

## Tests and verification

```bash
.venv/bin/python -m py_compile main.py models.py jev_client.py scripts/verify_cuda.py tests/test_scenarios.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python main.py --all
```

Tests cover loading, strict validation, the controlled personality experiment, request construction, action parsing, response distributions, optional metadata, CLI formatting/batching, panic threshold, configuration, unavailable server, timeout, HTTP errors and invalid response JSON. Mocked HTTP responses test the client contract only; they do not validate model quality. To evaluate actual behavior, run `--all` against the real server and inspect results without insisting on exact probabilistic answers.

Live verification completed on October 6, 2026, in Ubuntu 26.04.1 LTS under WSL2:

- **Both Linux environments: Python 3.12.15**, installed through uv 0.12.23. Ubuntu's system Python 3.14.4 was not used for the project.
- **Server: local-jev 0.2.0**, exact source revision `64a0b31ff343dca32142496cf9edca5a0174a18d`; **model: llm-qwen3.5-4b**, backed by `Qwen/Qwen3.5-4B`.
- Model weights were downloaded from Hugging Face and cached inside `.cache/huggingface`. The cached model snapshot is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- **PyTorch 2.14.1+cu130**, transformers 5.18.0. `torch.cuda.is_available()` returned **True**; PyTorch detected **NVIDIA GeForce RTX 5070**, compute capability **12.0**; an actual CUDA tensor operation returned **64.0**.
- The real model loaded and completed CUDA warm-up. The upstream backend automatically selects CUDA when available and moves its model and input tensors to that device. During real inference, `nvidia-smi` recorded **11,246 / 12,227 MiB** GPU memory used and **39%** GPU utilization. WSL did not expose per-process GPU entries; the empty process table is not evidence of CPU inference.
- First warm-up failed because Triton lacked a C compiler. After manual installation of Ubuntu `build-essential`, **GCC 15.2.0** compiled the required runtime code and startup succeeded. A `_POSIX_C_SOURCE` redefinition warning was nonfatal. No NVIDIA Linux display driver was installed.
- `GET /healthz` and `GET /v1/models` returned **HTTP 200**. A real direct `POST /v1/systemone` returned **HTTP 200**, both Choice and Score answers, model identity, probabilities, confidence, score legend and usage. Client parsing accepted the actual contract.
- Both requested commands completed with exit code **0**: `.venv/bin/python main.py scenarios/wounded_npc.json` and `.venv/bin/python main.py --all`.
- **26 tests passed again in Linux** after real inference. Syntax compilation and both environments' dependency checks passed.
- The ten recorded direct/integration requests took approximately **0.14-0.33 seconds each** after warm-up. This is a small observed run, not a performance benchmark. Cached startup still took several minutes on this mounted-drive setup.
- The working background server is bound to **http://127.0.0.1:8765**; startup helper, PID and logs are documented above. No further manual setup is required for the current installation.

### Actual results for the six bundled scenarios

| Scenario | Action | Danger | Selected action probability | Choice confidence |
|---|---|---:|---:|---:|
| Friendly player | negotiate | 0.0944 | 0.5640 | 0.4187 |
| Wounded NPC | flee | 0.8719 | 0.9435 | 0.9247 |
| Guards nearby | call_guards | 0.8820 | 0.4905 | 0.3207 |
| Aggressive NPC | attack | 0.6989 | 0.4498 | 0.2665 |
| Coward NPC | flee | 0.5195 | 0.9380 | 0.9174 |
| Diplomatic NPC | negotiate | 0.3144 | 0.9026 | 0.8702 |

All six returned `NORMAL MODE` because their danger scores were below the application's 0.9 panic threshold. The direct wounded request, single-scenario CLI and batch CLI agreed. Full terminal output and full action distributions are saved in `results/live/all-cli.txt` and each scenario's JSON file.

### Personality control with an identical display name

The bundled files differ in scenario name as well as personality, and names are part of the transmitted state. The verification script therefore repeated the experiment with the same name, `Personality comparison`, and identical physical world fields. Only personality and its description changed:

| Personality | Action | P(attack) | P(flee) | P(negotiate) | P(call_guards) | Danger |
|---|---|---:|---:|---:|---:|---:|
| aggressive | flee | 0.3979 | 0.4345 | 0.1139 | 0.0537 | 0.7769 |
| coward | flee | 0.0473 | 0.9322 | 0.0147 | 0.0058 | 0.6186 |
| diplomatic | negotiate | 0.0615 | 0.0273 | 0.8950 | 0.0161 | 0.2663 |

Personality changed the distributions and the diplomatic action in the controlled comparison. The aggressive attack/flee balance was close and its selected action flipped when the display name changed. This demonstrates semantic sensitivity, including sensitivity to incidental wording; it does not establish robust NPC behavior. The danger estimates also varied with personality despite equal physical conditions, so they should be treated as model rubric judgments, not objective risk measurements.

The complete audit trail is in ignored `results/live/`:

- `report.json`: readiness responses, direct request, all six raw scenario requests/responses, controlled personality requests, HTTP status, request IDs and timings.
- `direct-choice-score.json` and the six scenario JSON captures: actual API payloads and responses.
- `wounded-cli.txt`, `all-cli.txt`, `verification.txt`: actual terminal output.
- `cuda.txt`, `nvidia-smi.txt`, `server-requirements.txt`, `tests.txt`: environment and validation evidence.
- `server.log`, `server.pid`, `server-first-start.log`: successful startup state and the initial missing-compiler diagnosis.

## Troubleshooting and limitations

- **Cannot connect / timeout:** start the server, wait for loading, check `/healthz`, base URL and port, then increase `--timeout` for slow CPU inference. The client/server should run in the same WSL distribution.
- **HTTP 404:** check the model name and endpoint. **422:** check server API compatibility. **503:** inspect the server log for download, memory or CUDA failure. **401:** use the default server without `LOCAL_JEV_API_KEY`; this client does not implement optional server authentication.
- **Unexpected response:** the client rejects incompatible primitive types, unsupported actions, out-of-range scores and invalid probability distributions. Use the pinned upstream version.
- Small models can misinterpret context, depend on wording/option order and return confidently incorrect actions. This demo has no NPC animation, combat simulation or trained game-specific policy.
- Internet is needed for dependency and initial weight downloads; inference can run locally after caching. Keep the server bound to loopback.
- The live run demonstrated different personality distributions and successful CUDA inference. One small run does not establish model accuracy or robustness. Aggressive and guard-support choices had relatively low confidence, and the controlled display-name experiment exposed wording sensitivity.
- This machine's Linux environments and build source live under `/home/alex/.local/share/jev-runtime`, linked at the project paths. If moving the project to another machine, recreate those Linux environments. The preserved Windows environment is separate.
- The model consumes most of the available RTX 5070 memory alongside Windows applications. Large additional GPU workloads or longer inputs may cause memory exhaustion. Startup from the mounted Windows drive is slower than warmed-up requests.

## Project structure

```text
Jev/
├── README.md
├── requirements.txt
├── .gitignore
├── .python-version             # uv selects the Python 3.12 series
├── main.py
├── jev_client.py
├── models.py
├── scenarios/                  # six validated encounter JSON files
├── scripts/verify_cuda.py      # server PyTorch CUDA kernel check
├── scripts/verify_live.py      # save real HTTP responses and personality controls
├── scripts/start_server.sh     # optional background startup with PID and log
├── tests/test_scenarios.py
└── vendor/local-jev/           # inspected upstream clone, Git-ignored
```

Virtual environments, caches, model weights and generated results are ignored. No secrets or model weights are included in the project source.
