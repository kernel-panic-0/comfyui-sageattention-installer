# ComfyUI SageAttention Instataller

A bash script that installs [SageAttention](https://github.com/thu-ml/SageAttention) into an existing **ComfyUI** installation on **Ubuntu 24.04** (x86_64, NVIDIA GPU).

By default it installs **SageAttention 2**, compiled from source against your local CUDA toolkit. An interactive menu (or `--sa-version`) also offers **SageAttention 3** (FP4 kernels, Blackwell-only) and **SageAttention 1** from PyPI.

> **Why source-only for v2/v3?** PyPI's `sageattention` package only ships
> SageAttention **1.x** (latest 1.0.6). SageAttention 2 and 3 were never
> published to PyPI — they must be built from the `thu-ml/SageAttention`
> repository. The old "install the prebuilt wheel first" behaviour of this
> script silently installed SageAttention **1**; that path has been removed.

Both standard `pip`/`venv` setups and `uv`-managed ComfyUI installs are supported (auto-detected).

## Choosing a version

| Option          | What it is                                                                                                                              | Needs                                                                                                                                      |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `1`             | Triton-only kernels, PyPI wheel (`sageattention==1.0.6`)                                                                                | nothing extra                                                                                                                              |
| `2` *(default)* | CUDA kernels, built from source                                                                                                         | CUDA toolkit (nvcc) + C++ compiler                                                                                                         |
| `3`             | FP4 kernels ([sageattention3_blackwell](https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell)), built from source | **Blackwell GPU** (sm_100 / sm_120 / sm_121), CUDA ≥ 12.8, Python ≥ 3.13, torch ≥ 2.8, einops, internet (the build shallow-clones CUTLASS) |

SageAttention 3 installs as a **separate package** (`sageattn3`), so it coexists with v1/v2 rather than replacing them. The script refuses to build
v3 on a non-Blackwell GPU or with nvcc < 12.8 (upstream's `setup.py` rejects both anyway — this just fails fast with a clearer message).

## Prerequisites

Before running the script, make sure you have:

- **Ubuntu 24.04** on x86_64 (other distros may work but are untested).
- An **NVIDIA GPU** with working drivers (`nvidia-smi` runs).
  - Ampere (RTX 30), Ada (RTX 40), Hopper (H100) and Blackwell (RTX 50 / sm_120, B200 / sm_100) are recommended. Pre-Ampere cards will run but see little/no benefit.
- **ComfyUI already installed** (the script does not install ComfyUI itself), with a **CUDA build of PyTorch** (`torch>=2.3.0`) inside its environment.
- **Triton** (`triton>=3.0`) — the script will offer to install/upgrade it.
- **git** (`sudo apt install git`).
- For **SageAttention 2** (the default): a **CUDA toolkit (nvcc)** whose major version matches the CUDA your PyTorch was built against.
- For **SageAttention 3**: see the table above.
- **sudo** (optional) — only used to install missing system packages (e.g. `build-essential`) when you agree to the prompt.

### About the CUDA toolkit

SageAttention 2 and 3 must be compiled with `nvcc`. If no toolkit is found the script tells you and offers to install Ubuntu's package.

Be aware of a common Ubuntu 24.04 trap:

- `sudo apt install nvidia-cuda-toolkit` installs **CUDA 12.0.1** and puts `nvcc` at `/usr/bin/nvcc` (no `/usr/local/cuda` symlink). This is usually **too old** — it mismatches modern PyTorch builds, fails the per-arch CUDA floors (Ada fp8 ≥ 12.4, Hopper fp8 ≥ 12.3, Blackwell ≥ 12.8), and is **hard-rejected for SageAttention 3** (needs ≥ 12.8).
- For a matching toolkit, prefer NVIDIA's CUDA apt repo:
  `https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/`
  (install `cuda-keyring`, then e.g. `sudo apt install cuda-toolkit-13-0`), or use the official `.run` file from the CUDA Archive.

## Usage

```bash
chmod +x install_sageattention2.sh
./install_sageattention2.sh
```

The script is interactive by default: it asks which version to install (**2 is the default — just press Enter**), annotates whether your GPU supports SageAttention 3, then confirms the ComfyUI directory, the Python environment and any missing dependencies before building.

### Flags

```
      --comfyui-dir PATH   ComfyUI directory (default: ~/ComfyUI)
      --sa-version {1|2|3} SageAttention major version (default: 2)
                           2 = CUDA kernels, built from source (recommended)
                           3 = FP4 kernels, Blackwell only (sm_100/sm_120/
                               sm_121); needs CUDA >= 12.8, Python >= 3.13,
                               torch >= 2.8; installs the 'sageattn3' package
                           1 = Triton-only PyPI wheel (sageattention 1.0.6)
      --from-source        Kept for compatibility — v2 is always built from
                           source now
      --max-jobs N         Parallel compile jobs for the source build (default: nproc)
  -y, --non-interactive    Accept all defaults, never prompt
  -h, --help               Show help and exit
```

Environment: `CUDA_HOME` overrides toolkit detection; `CC`/`CXX` (both required) override host-compiler selection — otherwise the system `gcc`/`g++`
is used and snap-confined compilers are never chosen automatically.

Example — fully non-interactive, SageAttention 2:

```bash
./install_sageattention2.sh --comfyui-dir ~/ComfyUI -y
```

Example — SageAttention 3 on an RTX 50:

```bash
./install_sageattention2.sh --sa-version 3
```

Example — SageAttention 1 instead:

```bash
./install_sageattention2.sh --sa-version 1
```

Example — source build with 16 parallel jobs:

```bash
./install_sageattention2.sh --max-jobs 16
```

## What it does

1. **Locates ComfyUI** (flag value → prompt → default `~/ComfyUI`).
2. **Detects the Python environment** — `uv` if a `uv.lock` or `[tool.uv]` table is present, otherwise a standard venv (`.venv`/`venv` next to ComfyUI).
3. **Checks dependencies** — git, NVIDIA driver, Python ≥ 3.9, PyTorch ≥ 2.3.0 (CUDA build), Triton ≥ 3.0 (offers to install/upgrade), and detects your GPU's compute capability (used for the SageAttention 3 support check).
4. **Asks which version to install** (TUI menu, default 2) and shows whether your GPU is supported by SageAttention 3.
5. **Inspects the CUDA toolkit** (v2/v3 only) — looks for `nvcc` in `/usr/local/cuda-*`, `/usr/local/cuda`, `/usr` and `$CUDA_HOME`, warns about version mismatches and the per-arch CUDA floors (Blackwell sm_120 needs ≥ 12.8), and offers to install Ubuntu's package if nothing is found.
6. **Installs SageAttention**:
   - *v2/v3*: verifies a C++ compiler exists (offers `apt install build-essential`), installs the Python build deps (`ninja`, `setuptools`, `wheel`, `packaging`, `pybind11`), then runs a **toolchain probe** before the real build — a trivial C++ file compiled against torch's dev headers plus a one-line nvcc host-compiler check. This catches, up front: torch installs that ship no dev headers (stripped "portable" bundles — the script offers to reinstall PyTorch from the official wheel index), sandboxed compilers (e.g. snap clang cannot read `/mnt`, `/media` or host `/tmp`), and nvcc rejecting the host compiler (offers `gcc-12`). For v3 it additionally enforces the upstream gates (Blackwell GPU, nvcc ≥ 12.8) and checks `einops` before building. The build itself is a regular `pip install . --no-build-isolation --no-deps` (not editable) with
     `MAX_JOBS` and — for v2 — the README speed flags `EXT_PARALLEL=4`, `NVCC_APPEND_FLAGS="--threads 8"` (v3's setup.py tunes nvcc itself).
     v3 builds from the repo's `sageattention3_blackwell/` subdirectory. If the build still fails, the log is scanned for known missing-dependency
     patterns, the fix is offered, and the build can be retried (up to 3 attempts). The full log is kept in `sageattention_build.log` next to wherever you ran the script.
   - *v1*: `pip install sageattention==1.0.6` — no compilation.
7. **Runs a smoke test** — calls `sageattn(q, k, v)` (or `sageattn3_blackwell(q, k, v)` for v3) on a small CUDA tensor to confirm the kernel actually runs (not just that the import works).
8. **Prints a summary** and how to enable it in ComfyUI.

## Launching ComfyUI with SageAttention

```bash
cd ~/ComfyUI
# venv install:
./venv/bin/python main.py --use-sage-attention
# uv install:
uv run python main.py --use-sage-attention
```

**SageAttention 3 is different**: ComfyUI's core `--use-sage-attention` flag targets SageAttention 1/2. To use v3, select it per-model with the KJNodes *Patch Sage Attention KJ* node (`sageattn3_fp4_cuda` mode) or a dedicated SageAttention3 custom node:

```python
from sageattn3 import sageattn3_blackwell
attn_output = sageattn3_blackwell(q, k, v, is_causal=False)
```

If you use the **ComfyUI Desktop** app, don't pass the flag on the command line - enable SageAttention in the ComfyUI settings UI instead.

## Troubleshooting

**"SageAttention 3 supports Blackwell GPUs only".**
v3's FP4 kernels only exist for compute capabilities 10.0 (B200), 12.0 (RTX 50) and 12.1. On anything older, install SageAttention 2 instead.

**"SageAttention 3 requires a CUDA toolkit >= 12.8".**
Install a newer toolkit from NVIDIA's repo (see *About the CUDA toolkit* above). Note Ubuntu 24.04's `nvidia-cuda-toolkit` apt package (12.0.1) can never satisfy this.

**Build fails with `'torch/extension.h' file not found` (or `ATen/...`,
`pybind11/...`) even though the `-I` paths in the log look correct.**
Either your torch install ships no development headers — common with stripped-down "portable" torch bundles or environments copied between machines/drives (dangling symlinks) — or the compiler is sandboxed: snap clang cannot read `/mnt`, `/media` or the host `/tmp`, so every include under those paths fails. The script's toolchain probe detects both before building and offers to fix them (reinstall torch / switch to `/usr/bin/g++`). The script never auto-selects a compiler that resolves under `/snap`.

**nvcc fails with "unsupported GNU version" / "unsupported clang version".**
Your host compiler is newer than the CUDA toolkit supports. The script offers to install `gcc-12`/`g++-12` and use them as the host compiler.

**"No CUDA toolkit found" / nvcc errors.**
Your toolkit version probably doesn't match the CUDA PyTorch was built against. Check `nvcc --version` against the "CUDA" line in the script's summary, and install a matching toolkit from NVIDIA's repo (see *About the CUDA toolkit* above). Blackwell GPUs (RTX 50, compute capability 12.0) need CUDA ≥ 12.8.

**`ModuleNotFoundError: No module named 'triton'` when launching ComfyUI.**
Triton wasn't installed in the ComfyUI environment. Re-run the script (it will detect the missing Triton and offer to install it), or manually run `pip install triton` (≥ 3.0) in the same environment ComfyUI uses.

**Runtime smoke test fails after a successful build.**
Usually a CUDA/CUDA-toolkit mismatch or a too-old GPU. Re-check the toolkit version and your GPU compute capability against the floors in *Prerequisites*.

**uv environment not detected.**
The script looks for `uv.lock` or a `[tool.uv]` table in `pyproject.toml`. If your uv project is structured differently, run with `--comfyui-dir` pointing at the directory that contains those files.

**Spaces in the ComfyUI/venv path.**
The v1 wheel path handles spaces fine. The v2/v3 source builds do **not** support spaces in the venv/Python path — if you hit this, relocate your venv
to a space-free path, or use SageAttention 1.

## License

See [LICENSE](LICENSE).
