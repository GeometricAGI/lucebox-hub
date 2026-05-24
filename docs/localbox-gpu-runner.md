# Localbox GPU runner setup for C++ HTTP validation

This guide sets up a GPU-capable GitHub self-hosted runner ("localbox") that can run Lucebox's native C++ HTTP server validation. The validation entrypoint is `scripts/run_cpp_http_validation.sh`, which runs both the HTTP protocol probe and the benchmark suites against the native server.

## What this setup assumes

- Ubuntu 22.04 or newer on the runner host
- NVIDIA driver installed and `nvidia-smi` working
- CUDA toolkit installed with `nvcc` on `PATH`
- Python 3.12, CMake, Git, and `curl`
- Enough disk for the repo, build tree, and model weights
- A GitHub runner registration token for this repository or org

Recommended runner labels:

- `self-hosted`
- `linux`
- `x64`
- `gpu`
- `localbox`
- `cuda`

## 1. Prepare the host

Install the base toolchain on the GPU machine:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  curl \
  git \
  python3 \
  python3-venv
```

Install `uv` for workspace bootstrapping and model download helpers:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Confirm the GPU toolchain:

```bash
nvidia-smi
nvcc --version
python3 --version
cmake --version
```

## 2. Register the GitHub runner service

Get the registration token from GitHub before running `config.sh`:

1. Open `howard0su/lucebox-hub` on GitHub.
2. Go to **Settings** → **Actions** → **Runners**.
3. Click **New self-hosted runner**.
4. Choose the runner OS/architecture so GitHub shows the install commands.
5. Copy the temporary token from the generated `./config.sh --token ...` command.

The token is short-lived. If it expires, go back to the same page and generate a new one.

If you are registering the runner at the organization level instead, use **Organization settings** → **Actions** → **Runners** → **New self-hosted runner** and keep the same labels.

### Automate token refresh during provisioning

You do not need to renew the registration token for a runner that is already configured and running. The token is only needed when calling `config.sh`, such as first boot, rebuild, or replacement.

For automated provisioning, fetch a fresh token from the GitHub API right before registration instead of copying it manually:

```bash
OWNER=howard0su
REPO=lucebox-hub
GH_ADMIN_TOKEN=<repo-or-org-admin-token>

RUNNER_TOKEN="$(
  curl -fsSL -X POST \
    -H 'Accept: application/vnd.github+json' \
    -H "Authorization: Bearer $GH_ADMIN_TOKEN" \
    "https://api.github.com/repos/$OWNER/$REPO/actions/runners/registration-token" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])'
)"
```

Then use `"$RUNNER_TOKEN"` in `./config.sh --token ...`.

For an organization runner, call:

```text
POST https://api.github.com/orgs/<ORG>/actions/runners/registration-token
```

Best practice:

- keep the admin PAT or GitHub App credential in your bootstrap secret store, not on the runner image
- fetch the short-lived registration token just in time
- register the runner, then discard the token

On the localbox host:

```bash
mkdir -p /opt/actions-runner
cd /opt/actions-runner

curl -o actions-runner-linux-x64.tar.gz -L \
  https://github.com/actions/runner/releases/latest/download/actions-runner-linux-x64.tar.gz
tar xzf actions-runner-linux-x64.tar.gz

./config.sh \
  --url https://github.com/howard0su/lucebox-hub \
  --token <RUNNER_REGISTRATION_TOKEN> \
  --labels gpu,localbox,cuda \
  --unattended

sudo ./svc.sh install
sudo ./svc.sh start
```

If you register the runner at the organization level, keep the same labels so the workflow can target this machine explicitly.

## 3. Bootstrap the repository on the runner

Use a stable workspace path so model weights and build artifacts can be reused between jobs:

```bash
mkdir -p /srv/lucebox
cd /srv/lucebox
git clone --recurse-submodules https://github.com/howard0su/lucebox-hub.git
cd lucebox-hub

uv sync --frozen
cmake -B dflash/build -S dflash -DCMAKE_BUILD_TYPE=Release
cmake --build dflash/build --target dflash_server -j"$(nproc)"
```

If the GPU is fixed to one architecture, you can shrink build time with:

```bash
cmake -B dflash/build -S dflash \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
```

## 4. Prime model files

The validation script expects the native server target and draft weights to exist already.

```bash
uv run hf download unsloth/Qwen3.6-27B-GGUF \
  Qwen3.6-27B-Q4_K_M.gguf \
  --local-dir dflash/models/

uv run hf download Lucebox/Qwen3.6-27B-DFlash-GGUF \
  dflash-draft-3.6-q8_0.gguf \
  --local-dir dflash/models/draft/
```

If you store weights somewhere else on the runner, pass `TARGET=` and `DRAFT=` to the validation script.

## 5. Run the validation locally on the runner

From the repository root:

```bash
bash scripts/run_cpp_http_validation.sh
```

Useful overrides:

```bash
TARGET=/models/Qwen3.6-27B-Q4_K_M.gguf \
DRAFT=/models/draft/dflash-draft-3.6-q8_0.gguf \
CLIENTS=all \
MAX_CTX=32768 \
MAX_TOKENS=512 \
BUDGET=22 \
VERIFY_MODE=ddtree \
FA_WINDOW=2048 \
bash scripts/run_cpp_http_validation.sh
```

Artifacts land under `.artifacts/cpp-http-validation/<timestamp>/`:

- `dflash_server.log`
- `probe.json`
- `bench.json`

Fast-CI override example:

```bash
BENCH_SUITE=he,math \
BENCH_N_SAMPLE=3 \
bash scripts/run_cpp_http_validation.sh
```

## 6. Call it from GitHub Actions

Example job:

```yaml
gpu-http-validation:
  name: GPU HTTP validation (C++ server)
  runs-on: [self-hosted, linux, x64, gpu, localbox]
  timeout-minutes: 90

  steps:
    - uses: actions/checkout@v4
      with:
        submodules: recursive

    - uses: astral-sh/setup-uv@v3
      with:
        version: "0.11.x"

    - name: Sync workspace
      run: uv sync --frozen

    - name: Validate native C++ server
      env:
        TARGET: /srv/models/Qwen3.6-27B-Q4_K_M.gguf
        DRAFT: /srv/models/draft/dflash-draft-3.6-q8_0.gguf
        CMAKE_CUDA_ARCHITECTURES: "86"
        CLIENTS: all
        BENCH_SUITE: he,math,agent
        BENCH_N_SAMPLE: "3"
        MAX_CTX: "32768"
        MAX_TOKENS: "512"
        BUDGET: "22"
        VERIFY_MODE: ddtree
        FA_WINDOW: "2048"
      run: bash scripts/run_cpp_http_validation.sh

    - name: Upload validation artifacts
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: cpp-http-validation
        path: .artifacts/cpp-http-validation/
```

## 7. Troubleshooting

- If the job fails before probing or during bench, check `dflash_server.log` for model-path, CUDA, or OOM errors.
- If the machine already has a fresh binary, set `BUILD_SERVER=0` to skip the compile step.
- If startup is slow on a cold machine, increase `START_TIMEOUT`.
- If you only want the protocol coverage, set `RUN_BENCH=0`.
- If you only want to validate the protocol surface and not install client packages, keep the defaults; the harness probe is HTTP-only unless `INSTALL_PACKAGES=1` or `PACKAGE_SMOKE=1` is set.
