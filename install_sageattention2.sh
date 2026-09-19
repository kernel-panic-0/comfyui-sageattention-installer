#!/usr/bin/env bash
# =============================================================================
#  install_sageattention2.sh
#  Install SageAttention for ComfyUI on Ubuntu 24.04
#
#  Strategy:
#    - SageAttention 2 (DEFAULT): built from source against the local CUDA
#      toolkit. PyPI's 'sageattention' package only ships SageAttention 1.x,
#      so v2 must be compiled from the thu-ml/SageAttention repo. (The old
#      "pip install sageattention" wheel-first path silently installed
#      SageAttention 1 — that path has been removed.)
#    - SageAttention 3 (optional, Blackwell only): FP4 kernels from the
#      sageattention3_blackwell subdirectory of the same repo. Installs a
#      SEPARATE distribution ('sageattn3') that coexists with v1/v2.
#      Requires a Blackwell GPU (sm_100/sm_120/sm_121), CUDA >= 12.8,
#      Python >= 3.13 and torch >= 2.8 per the upstream README.
#    - SageAttention 1 (optional, chosen from the interactive menu or
#      --sa-version 1): plain PyPI wheel sageattention==1.0.6 (Triton only).
#    - Before compiling, a toolchain probe verifies that the chosen C++
#      compiler can actually see torch's dev headers (catches stripped torch
#      bundles and snap-confined compilers) and that nvcc accepts the host
#      compiler. Missing dependencies trigger a notify-and-prompt offer to
#      install them.
#    - If the build still fails, the log is scanned for known
#      missing-dependency patterns and the user is offered a fix + retry.
#
#  Supports:
#    - Standard pip/venv installs
#    - uv-managed environments (passively detected)
#
#  Requirements (see README.md for detail):
#    - Ubuntu 24.04 on x86_64
#    - NVIDIA GPU (Ampere / Ada / Hopper / Blackwell recommended)
#    - ComfyUI already installed, with a CUDA build of PyTorch >= 2.3.0
#    - git
#    - For SageAttention 2: a CUDA toolkit (nvcc) matching PyTorch's CUDA
#    - sudo (only if system packages like build-essential are missing)
#
#  Usage:
#    chmod +x install_sageattention2.sh
#    ./install_sageattention2.sh                     # interactive, v2 default
#    ./install_sageattention2.sh --sa-version 3      # SageAttention 3 (Blackwell)
#    ./install_sageattention2.sh --sa-version 1      # SageAttention 1 (PyPI)
#    ./install_sageattention2.sh --comfyui-dir PATH  # explicit ComfyUI path
#    ./install_sageattention2.sh -y                  # accept all defaults
# =============================================================================

set -euo pipefail

# ── Colour setup (TTY-gated so logs stay clean) ──────────────────────────────
# Colour vars hold real escape bytes (ANSI-C $'...' quoting) so they can be
# embedded in messages and printed via %s without backslash re-interpretation.
if [[ -t 1 ]]; then
    RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
    CYAN=$'\033[0;36m'; BOLD=$'\033[1m'; NC=$'\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; NC=''
fi

info()    { printf '%s\n' "${CYAN}[INFO]${NC}  $*"; }
success() { printf '%s\n' "${GREEN}[OK]${NC}    $*"; }
warn()    { printf '%s\n' "${YELLOW}[WARN]${NC}  $*" >&2; }
error()   { printf '%s\n' "${RED}[ERROR]${NC} $*" >&2; }
die()     { error "$*"; exit 1; }
header()  { printf '\n%s==> %s%s\n' "${BOLD}${CYAN}" "$*" "${NC}"; }

# ── read wrapper: dies with a clear message on EOF/Ctrl-D instead of letting
#    errexit kill the script silently (fixes silent abort under non-TTY stdin).
# Usage: ask "prompt text" dest_var
# The destination is taken by nameref so shellcheck can follow it.
ask() {
    local prompt="$1"
    local -n _ask_dest="$2"
    if ! read -r -p "$prompt" _ask_dest; then
        _ask_dest=''
        echo "" >&2
        die "End of input on stdin — re-run in a terminal or pass the relevant --flag (see --help)."
    fi
}

# ── confirm: yes/no question with a default.
# Usage: confirm "Question?" y|n   → return 0 = yes, 1 = no
# In non-interactive mode the default is taken without prompting.
confirm() {
    local question="$1" default="${2:-y}" answer
    if $NON_INTERACTIVE; then
        [[ "$default" == "y" ]] && return 0
        return 1
    fi
    if [[ "$default" == "y" ]]; then
        ask "$question [Y/n]: " answer
        answer="${answer:-y}"
    else
        ask "$question [y/N]: " answer
        answer="${answer:-n}"
    fi
    [[ "$answer" =~ ^[Yy]$ ]]
}

# ── Argument parsing ─────────────────────────────────────────────────────────
FROM_SOURCE=false       # kept for compatibility: v2 is now ALWAYS source-built
SA_VERSION_ARG=""       # "" (ask in TUI, default 2) | "1" | "2"
COMFYUI_DIR_ARG=""
MAX_JOBS_ARG=""
NON_INTERACTIVE=false
PRINT_HELP=false

show_help() {
    cat <<'HELP'
install_sageattention2.sh — Install SageAttention 1 or 2 for ComfyUI (Ubuntu 24.04)

Usage:
  ./install_sageattention2.sh [options]

Options:
      --comfyui-dir PATH   ComfyUI directory (default: ~/ComfyUI)
      --sa-version {1|2|3} SageAttention major version (default: 2)
                           2 = CUDA kernels, built from source (recommended)
                           3 = FP4 kernels, Blackwell only (sm_100/sm_120/
                               sm_121); needs CUDA >= 12.8, Python >= 3.13,
                               torch >= 2.8; installs the 'sageattn3' package
                           1 = Triton-only PyPI wheel (sageattention 1.0.6)
      --from-source        Kept for compatibility — SageAttention 2 is always
                           built from source now (PyPI only ships v1)
      --max-jobs N         Parallel compile jobs for the source build (default: nproc)
  -y, --non-interactive    Accept all defaults, never prompt
  -h, --help               Show this help and exit

Environment:
  CUDA_HOME   If set and pointing at a valid toolkit, used for the source build.
  CC / CXX    If both are set they are honoured as the host compiler (the
              default is the system gcc/g++; snap-confined compilers are
              never selected automatically).

Most users need no flags — the script asks which version to install (default:
SageAttention 2 from source) and auto-detects the ComfyUI environment.
HELP
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from-source)        FROM_SOURCE=true; shift ;;
        --comfyui-dir)        COMFYUI_DIR_ARG="${2:-}"; shift 2 ;;
        --comfyui-dir=*)      COMFYUI_DIR_ARG="${1#*=}"; shift ;;
        --sa-version)         SA_VERSION_ARG="${2:-}"; shift 2 ;;
        --sa-version=*)       SA_VERSION_ARG="${1#*=}"; shift ;;
        --max-jobs)           MAX_JOBS_ARG="${2:-}"; shift 2 ;;
        --max-jobs=*)         MAX_JOBS_ARG="${1#*=}"; shift ;;
        -y|--non-interactive) NON_INTERACTIVE=true; shift ;;
        -h|--help)            PRINT_HELP=true; shift ;;
        *) die "Unknown argument: '$1' (try --help)" ;;
    esac
done

$PRINT_HELP && { show_help; exit 0; }

if [[ -n "$SA_VERSION_ARG" ]] && [[ ! "$SA_VERSION_ARG" =~ ^[123]$ ]]; then
    die "--sa-version must be 1, 2 or 3 (got '$SA_VERSION_ARG')."
fi

# ── Banner ────────────────────────────────────────────────────────────────────
printf '%s' "$BOLD"
cat <<'BANNER'
  ╔══════════════════════════════════════════════════╗
  ║        SageAttention Installer for ComfyUI       ║
  ║    Ubuntu 24.04  •  v2 / v3 (Blackwell) / v1     ║
  ╚══════════════════════════════════════════════════╝
BANNER
printf '%s\n' "$NC"

# ── APT helper: update once, install packages, surface real errors ──────────
APT_UPDATED=false
apt_install() {
    # apt_install PKG [PKG ...]
    if ! command -v sudo &>/dev/null; then
        warn "sudo is not available — cannot install: $*"
        return 1
    fi
    if ! $APT_UPDATED; then
        info "Running 'sudo apt-get update' (once)..."
        if ! sudo apt-get update -qq; then
            warn "'apt-get update' failed. Package installs may fail too."
        fi
        APT_UPDATED=true
    fi
    sudo apt-get install -y "$@"
}

# ── 1. Locate ComfyUI ─────────────────────────────────────────────────────────
header "Step 1 of 8 – Locate ComfyUI installation"

COMFYUI_DEFAULT="$HOME/ComfyUI"
if [[ -n "$COMFYUI_DIR_ARG" ]]; then
    COMFYUI_DIR="$COMFYUI_DIR_ARG"
elif $NON_INTERACTIVE; then
    COMFYUI_DIR="$COMFYUI_DEFAULT"
else
    printf 'Where is your ComfyUI directory?\n'
    printf '  Press %sEnter%s to accept the default: %s%s%s\n' "$BOLD" "$NC" "$CYAN" "$COMFYUI_DEFAULT" "$NC"
    ask "  Path: " COMFYUI_DIR
    COMFYUI_DIR="${COMFYUI_DIR:-$COMFYUI_DEFAULT}"
fi
COMFYUI_DIR="${COMFYUI_DIR/#\~/$HOME}"
[[ -d "$COMFYUI_DIR" ]] || die "Directory '$COMFYUI_DIR' does not exist."
success "ComfyUI directory: $COMFYUI_DIR"

# ── 2. Detect Python environment (PASSIVE — never invokes uv run as a probe) ─
header "Step 2 of 8 – Detect Python environment"

UV_BIN=""
PYTHON_BIN=""
ENV_TYPE=""   # "uv" | "venv" | "manual"

# Find uv on PATH or its usual install locations, but do NOT run it yet.
for uv_candidate in "uv" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if command -v "$uv_candidate" &>/dev/null; then
        UV_BIN=$(command -v "$uv_candidate")
        break
    fi
done

# A uv *project* is identified by passive markers only.
UV_PROJECT=false
if [[ -n "$UV_BIN" ]]; then
    if [[ -f "$COMFYUI_DIR/uv.lock" ]]; then
        UV_PROJECT=true
    elif [[ -f "$COMFYUI_DIR/pyproject.toml" ]] \
            && grep -Eq '^[[:space:]]*\[[[:space:]]*tool[[:space:]]*\.[[:space:]]*u' "$COMFYUI_DIR/pyproject.toml"; then
        # pyproject.toml has a [tool.uv] table -> uv-managed
        UV_PROJECT=true
    fi
fi

if $UV_PROJECT; then
    info "uv detected at: $UV_BIN"
    info "ComfyUI appears to be a uv-managed project (uv.lock or [tool.uv] found)."
    printf '  uv manages its own Python and dependencies. The installer will use\n'
    printf '  %suv run%s to install/build SageAttention in the correct environment.\n\n' "$CYAN" "$NC"
    # Now it's legitimate to actually run uv to resolve the python interpreter.
    PYTHON_BIN=$("$UV_BIN" run --project "$COMFYUI_DIR" python -c "import sys; print(sys.executable)")
    ENV_TYPE="uv"
    success "uv Python: $PYTHON_BIN"
else
    # ── Standard venv auto-detect ─────────────────────────────────────────────
    DETECTED_PYTHON=""
    for candidate in \
        "$COMFYUI_DIR/.venv/bin/python" \
        "$COMFYUI_DIR/venv/bin/python" \
        "$COMFYUI_DIR/../venv/bin/python"
    do
        if [[ -x "$candidate" ]]; then
            DETECTED_PYTHON="$candidate"
            break
        fi
    done

    if [[ -n "$DETECTED_PYTHON" ]]; then
        printf '  Auto-detected Python: %s%s%s\n' "$CYAN" "$DETECTED_PYTHON" "$NC"
        if $NON_INTERACTIVE; then
            USE_DETECTED="Y"
        else
            ask "  Use this? [Y/n]: " USE_DETECTED
        fi
        if [[ "${USE_DETECTED:-Y}" =~ ^[Yy]$ ]]; then
            PYTHON_BIN="$DETECTED_PYTHON"
        else
            DETECTED_PYTHON=""
        fi
    fi

    if [[ -z "$DETECTED_PYTHON" ]]; then
        if $NON_INTERACTIVE; then
            die "Could not auto-detect ComfyUI's Python. Re-run with --comfyui-dir pointing at a tree containing .venv/bin/python or venv/bin/python."
        fi
        printf '  Enter the path to the Python binary used by ComfyUI.\n'
        printf '  Examples:\n'
        printf '    %s~/ComfyUI/venv/bin/python%s   (typical pip install)\n' "$CYAN" "$NC"
        printf '    %s/usr/bin/python3%s             (system Python)\n' "$CYAN" "$NC"
        ask "  Python path: " PYTHON_BIN
        PYTHON_BIN="${PYTHON_BIN/#\~/$HOME}"
    fi

    [[ -x "$PYTHON_BIN" ]] || die "Python binary '$PYTHON_BIN' not found or not executable."
    ENV_TYPE="venv"
fi

PYTHON_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) \
    || die "Could not query Python version — is '$PYTHON_BIN' a working Python?"
PY_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
(( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 9) )) && \
    die "SageAttention 2 requires Python >= 3.9 (found $PYTHON_VERSION)."
success "Python $PYTHON_VERSION  ($ENV_TYPE)"

# ── Helper: run a command inside the right environment ────────────────────────
# Usage: pyrun <python args...>     e.g. pyrun -m pip install foo
#                                     pyrun -c "import torch"
pyrun() {
    if [[ "$ENV_TYPE" == "uv" ]]; then
        "$UV_BIN" run --project "$COMFYUI_DIR" python "$@"
    else
        "$PYTHON_BIN" "$@"
    fi
}

# ── Helper: read a distribution version via importlib.metadata (robust) ──────
# dist_version MODULE_NAME  -> echoes version or "not_installed"
dist_version() {
    pyrun -c "
import importlib.metadata as m, sys
try:
    print(m.version('$1'))
except Exception:
    print('not_installed')
" 2>/dev/null || echo "not_installed"
}

# ── 3. System dependency checks ───────────────────────────────────────────────
header "Step 3 of 8 – Checking system dependencies"

# --- git (the script depends on it for clone/update) ---
command -v git &>/dev/null || die "git not found. Install with: sudo apt install git"
success "git: $(git --version)"

# --- NVIDIA driver ---
command -v nvidia-smi &>/dev/null || die "nvidia-smi not found. Install NVIDIA drivers first."
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
success "NVIDIA driver: $DRIVER_VER"

# --- PyTorch: present AND meets the >=2.3.0 floor ---
TORCH_VER=$(dist_version torch)
[[ "$TORCH_VER" != "not_installed" ]] || \
    die "PyTorch is not installed in the ComfyUI environment. Install PyTorch first."
TORCH_OK=$(pyrun -c "
import torch, re
def vt(s): return tuple(int(x) for x in re.findall(r'\d+', s.split('+')[0]))
print('ok' if vt(torch.__version__) >= (2, 3, 0) else 'old')
" 2>/dev/null || echo "old")
if [[ "$TORCH_OK" != "ok" ]]; then
    die "SageAttention 2 requires torch>=2.3.0 (found $TORCH_VER). Upgrade PyTorch in your ComfyUI environment."
fi
success "PyTorch: $TORCH_VER"

TORCH_CUDA_FOR_BUILD=$(pyrun -c "import torch; v=torch.version.cuda; print(v if v else '')" 2>/dev/null || true)
[[ -n "$TORCH_CUDA_FOR_BUILD" ]] || \
    die "Could not read torch.version.cuda. Is this a CPU-only PyTorch? SageAttention needs a CUDA build."
info "PyTorch built against CUDA $TORCH_CUDA_FOR_BUILD"

# --- Triton: detect; honour the user's "no" (this was a real bug) ---
TRITON_VER=$(dist_version triton)
if [[ "$TRITON_VER" == "not_installed" ]]; then
    printf '  Triton is not installed. SageAttention requires triton >= 3.0.\n'
    if confirm "  Install now?" y; then
        if pyrun -m pip install triton --quiet; then
            TRITON_VER=$(dist_version triton)
            success "Triton installed: $TRITON_VER"
        else
            warn "Triton install failed — install it manually with: pip install triton"
        fi
    else
        warn "Skipping Triton — SageAttention import/build will fail without it."
    fi
else
    TRITON_MAJOR=$(echo "$TRITON_VER" | cut -d. -f1)
    if (( TRITON_MAJOR < 3 )); then
        warn "Triton $TRITON_VER < 3.0."
        if confirm "  Upgrade?" y; then
            if pyrun -m pip install --upgrade triton --quiet; then
                TRITON_VER=$(dist_version triton)
                success "Triton upgraded: $TRITON_VER"
            else
                warn "Triton upgrade failed — continuing with $TRITON_VER."
            fi
        else
            warn "Keeping Triton $TRITON_VER — build may fail."
        fi
    else
        success "Triton: $TRITON_VER"
    fi
fi

# --- GPU compute capability (needed for the version menu, the CUDA-floor
#     warnings and the SageAttention 3 Blackwell gate) ---
# Guarded: a torch install whose metadata exists but which fails to import
# must not abort the script here.
CC_RAW=$(pyrun - 2>/dev/null <<'EOF' || echo "unknown"
import torch
try:
    ccs = set()
    for i in range(torch.cuda.device_count()):
        maj, mi = torch.cuda.get_device_capability(i)
        ccs.add(f"{maj}.{mi}")
    print(",".join(sorted(ccs)))
except Exception:
    print("unknown")
EOF
)
SA3_SUPPORTED=false
if [[ "$CC_RAW" == "unknown" ]]; then
    warn "Could not detect GPU compute capability."
else
    success "GPU compute capability: $CC_RAW"
    for cc in $(echo "$CC_RAW" | tr ',' ' '); do
        case "$cc" in
            10.0|12.0|12.1) SA3_SUPPORTED=true ;;
        esac
    done
fi

# ── 4. Choose the SageAttention version (TUI) ────────────────────────────────
header "Step 4 of 8 – Choose SageAttention version"

printf '  Available SageAttention generations:\n\n'
printf '    %s[1]%s  SageAttention 1  — Triton kernels only\n' "$CYAN" "$NC"
printf '          installed from PyPI (sageattention==1.0.6)\n'
printf '          needs: no CUDA toolkit, no compiler\n\n'
printf '    %s[2]%s  SageAttention 2  — CUDA kernels, best supported\n' "$CYAN" "$NC"
printf '          built from source (thu-ml/SageAttention)\n'
printf '          needs: CUDA toolkit (nvcc) + a C++ compiler\n\n'
printf '    %s[3]%s  SageAttention 3  — FP4 kernels, %sBlackwell only%s\n' "$CYAN" "$NC" "$BOLD" "$NC"
printf '          built from source (sageattention3_blackwell)\n'
printf '          needs: Blackwell GPU (RTX 50 / sm_120, B200 / sm_100),\n'
printf '                 CUDA >= 12.8, Python >= 3.13, torch >= 2.8\n\n'
printf '  Note: PyPI only ships SageAttention 1.x — versions 2 and 3 must be\n'
printf '  compiled. SageAttention 3 installs as a separate package (sageattn3)\n'
printf '  and coexists with v1/v2.\n\n'

if $SA3_SUPPORTED; then
    printf '  Your GPU (compute capability %s) %sis%s supported by SageAttention 3.\n\n' "$CC_RAW" "$BOLD" "$NC"
elif [[ "$CC_RAW" != "unknown" ]]; then
    printf '  Your GPU (compute capability %s) is %snot%s supported by SageAttention 3\n' "$CC_RAW" "$BOLD" "$NC"
    printf '  (needs 10.0 / 12.0 / 12.1) — option 3 will refuse to run.\n\n'
fi

if $NON_INTERACTIVE; then
    SA_MAJOR="${SA_VERSION_ARG:-2}"
else
    while true; do
        ask "  Which version do you want to install? [${SA_VERSION_ARG:-2}]: " SA_CHOICE
        SA_CHOICE="${SA_CHOICE:-${SA_VERSION_ARG:-2}}"
        case "$SA_CHOICE" in
            1) SA_MAJOR=1; break ;;
            2) SA_MAJOR=2; break ;;
            3) SA_MAJOR=3; break ;;
            q|Q) die "Cancelled at user request." ;;
            *) warn "Please answer 1, 2, 3, or q to cancel." ;;
        esac
    done
fi

case "$SA_MAJOR" in
    3)
        success "Selected: SageAttention 3 (Blackwell FP4 source build)."
        SA_METHOD="Blackwell FP4 source build"
        ;;
    2)
        success "Selected: SageAttention 2 (source build)."
        SA_METHOD="source build"
        ;;
    *)
        success "Selected: SageAttention 1 (PyPI wheel)."
        SA_METHOD="PyPI wheel"
        ;;
esac

# ── 5. CUDA toolkit detection (only needed for the SageAttention 2/3 source
#    builds, but surfaced early so the user gets a clear message first) ───────
header "Step 5 of 8 – CUDA toolkit (for the source build)"

NVCC_VER="unknown"
CUDA_HOME_FOUND=""
NVCC_FROM_APT=false

if [[ "$SA_MAJOR" == "1" ]]; then
    info "Skipped — SageAttention 1 does not need a CUDA toolkit."
else
    # Look in the usual locations AND /usr, where Ubuntu's nvidia-cuda-toolkit
    # installs /usr/bin/nvcc without creating /usr/local/cuda.
    detect_cuda_toolkit() {
        CUDA_HOME_FOUND=""
        NVCC_FROM_APT=false
        local candidate
        for candidate in \
            "/usr/local/cuda-${TORCH_CUDA_FOR_BUILD}" \
            "/usr/local/cuda" \
            "/usr"
        do
            if [[ -x "${candidate}/bin/nvcc" ]]; then
                CUDA_HOME_FOUND="$candidate"
                if [[ "$candidate" == "/usr" ]]; then
                    NVCC_FROM_APT=true
                fi
                return 0
            fi
        done
        # Allow an explicit CUDA_HOME from the environment to win if it's valid.
        if [[ -n "${CUDA_HOME:-}" ]] && [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
            CUDA_HOME_FOUND="$CUDA_HOME"
            return 0
        fi
        return 1
    }

    if ! detect_cuda_toolkit; then
        warn "No CUDA toolkit found (looked in /usr/local/cuda-${TORCH_CUDA_FOR_BUILD}, /usr/local/cuda, /usr, \$CUDA_HOME)."
        printf '  SageAttention %s must be compiled with nvcc, and the toolkit version\n' "$SA_MAJOR"
        printf '  should match PyTorch'"'"'s CUDA (%s).\n' "$TORCH_CUDA_FOR_BUILD"
        printf '  Install options:\n'
        printf '    %ssudo apt install nvidia-cuda-toolkit%s  (Ubuntu 24.04 → CUDA 12.0; usually too old)\n' "$CYAN" "$NC"
        printf '    NVIDIA repo: %shttps://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/%s\n' "$CYAN" "$NC"
        printf '    (install cuda-keyring, then e.g. sudo apt install cuda-toolkit-%s-%s)\n\n' \
            "$(echo "$TORCH_CUDA_FOR_BUILD" | cut -d. -f1)" "$(echo "$TORCH_CUDA_FOR_BUILD" | cut -d. -f2)"
        if confirm "  Attempt 'sudo apt install nvidia-cuda-toolkit' now anyway?" n; then
            apt_install nvidia-cuda-toolkit || true
            detect_cuda_toolkit || warn "Toolkit still not detected after apt install."
        fi
    fi

    if [[ -n "$CUDA_HOME_FOUND" ]]; then
        export CUDA_HOME="$CUDA_HOME_FOUND"
        NVCC_VER=$("${CUDA_HOME}/bin/nvcc" --version | grep -oP 'release \K[0-9.]+')
        if $NVCC_FROM_APT; then
            warn "nvcc found at /usr/bin/nvcc (Ubuntu 'nvidia-cuda-toolkit', CUDA $NVCC_VER)."
            printf "  Ubuntu 24.04's apt package ships CUDA %s12.0.1%s, which is older than what\n" "$BOLD" "$NC"
            printf "  PyTorch is built against (%s) and %sfails%s the per-arch floors below.\n" "$TORCH_CUDA_FOR_BUILD" "$BOLD" "$NC"
            printf "  If the source build fails, install a matching toolkit from NVIDIA's repo:\n"
            printf "    %shttps://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/%s\n" "$CYAN" "$NC"
            printf "  (cuda-keyring, then: sudo apt install cuda-toolkit-12-x)\n"
        else
            success "CUDA_HOME: $CUDA_HOME  (nvcc $NVCC_VER)"
        fi
        NVCC_MAJOR=$(echo "$NVCC_VER" | cut -d. -f1)
        TORCH_MAJOR_C=$(echo "$TORCH_CUDA_FOR_BUILD" | cut -d. -f1)
        (( NVCC_MAJOR != TORCH_MAJOR_C )) && \
            warn "nvcc major version ($NVCC_VER) differs from PyTorch CUDA ($TORCH_CUDA_FOR_BUILD). Build may fail."
    else
        warn "Proceeding without a CUDA toolkit — the source build in Step 6 will stop with guidance."
    fi

    # --- Per-arch CUDA floor warnings (CC itself was detected in Step 3) ---
    declare -a ARCH_LIST=()
    if [[ "$CC_RAW" == "unknown" ]]; then
        :   # already warned in Step 3
    else
        # Per-arch CUDA floors from the SageAttention README.
        for cc in $(echo "$CC_RAW" | tr ',' ' '); do
            ARCH_LIST+=("$cc")
            cc_major=${cc%%.*}
            case "$cc_major" in
                8)   floor="12.0" ; label="Ampere"  ;;
                9)   floor="12.3" ; label="Hopper (fp8)" ;;
                10)  floor="12.8" ; label="Blackwell (sm_100)" ;;
                12)  floor="12.8" ; label="Blackwell (sm_120, RTX 50)" ;;
                *)   floor=""    ; label="" ;;
            esac
            # Ada (sm_89) is 8.x major but 12.4 floor for fp8 — handle explicitly.
            if [[ "$cc" == "8.9" ]]; then floor="12.4"; label="Ada (fp8)"; fi
            if (( cc_major < 8 )); then
                warn "Compute capability $cc is pre-Ampere; performance gains will be limited."
            fi
            if [[ -n "$floor" && "$NVCC_VER" != "unknown" ]]; then
                # Compare NVCC_VER (e.g. 12.0.1) against floor (e.g. 12.3) numerically.
                nvcc_major_minor=$(echo "$NVCC_VER" | awk -F. '{printf "%d.%d\n", $1, $2}')
                if ! pyrun -c "
import re
def vt(s): return tuple(int(x) for x in re.findall(r'\d+', s))
exit(0 if vt('$nvcc_major_minor') >= vt('$floor') else 1)
" 2>/dev/null; then
                    warn "nvcc $NVCC_VER is below the $label CUDA floor ($floor). fp8 / 2++ kernels may fail to build."
                fi
            fi
        done
    fi
fi

# ── 6. Install SageAttention ─────────────────────────────────────────────────
header "Step 6 of 8 – Install SageAttention"

# v1 and v2 share the 'sageattention' distribution; v3 is the separate
# 'sageattn3' package and coexists with whichever of v1/v2 is installed.
if [[ "$SA_MAJOR" == "3" ]]; then
    SA_DIST_NAME="sageattn3"
else
    SA_DIST_NAME="sageattention"
fi

SA_VER=$(dist_version "$SA_DIST_NAME")

DO_INSTALL=true
if [[ "$SA_VER" != "not_installed" ]]; then
    printf '  %s %s%s%s is already installed.\n' "$SA_DIST_NAME" "$CYAN" "$SA_VER" "$NC"
    SA_INSTALLED_MAJOR=$(printf '%s' "$SA_VER" | cut -d. -f1)
    if [[ "$SA_MAJOR" == "3" || "$SA_INSTALLED_MAJOR" == "$SA_MAJOR" ]]; then
        if ! confirm "  Reinstall / upgrade it?" n; then
            DO_INSTALL=false
        fi
    else
        printf '  You selected SageAttention %s — this will %sreplace%s the v%s install.\n' \
            "$SA_MAJOR" "$BOLD" "$NC" "$SA_INSTALLED_MAJOR"
        confirm "  Continue?" y || die "Aborting at user request."
    fi
fi

if ! $DO_INSTALL; then
    success "Keeping the existing $SA_DIST_NAME $SA_VER."
else
    # ════════════════════════════════════════════════════════════════════════
    #  SageAttention 1 — PyPI wheel (no compilation)
    # ════════════════════════════════════════════════════════════════════════
    if [[ "$SA_MAJOR" == "1" ]]; then
        header "Step 6b – Installing SageAttention 1 from PyPI"
        info "Installing sageattention==1.0.6 (Triton-only SageAttention 1)..."
        if ! pyrun -m pip install "sageattention==1.0.6"; then
            die "Failed to install sageattention 1.0.6 from PyPI."
        fi
        SA_VER_NEW=$(dist_version sageattention)
        [[ "$SA_VER_NEW" != "not_installed" ]] || die "sageattention did not install correctly."
        success "SageAttention $SA_VER_NEW installed from PyPI."

    # ════════════════════════════════════════════════════════════════════════
    #  SageAttention 2 / 3 — source build
    # ════════════════════════════════════════════════════════════════════════
    else
        header "Step 6b – Building SageAttention $SA_MAJOR from source"
        $FROM_SOURCE && [[ "$SA_MAJOR" == "2" ]] && info "--from-source noted: v2 is always built from source now."

        # Validate we actually have a CUDA toolkit before doing real work.
        if [[ -z "$CUDA_HOME_FOUND" || "$NVCC_VER" == "unknown" ]]; then
            error "A CUDA toolkit (nvcc) is required to build SageAttention $SA_MAJOR."
            printf '  Install one matching your PyTorch CUDA version (%s), e.g. from\n' "$TORCH_CUDA_FOR_BUILD"
            printf '  %shttps://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/%s\n' "$CYAN" "$NC"
            die  "Re-run this script after installing the toolkit."
        fi

        # ── SageAttention 3 extra gates ──────────────────────────────────────
        # setup.py itself refuses non-Blackwell GPUs ("Unsupported GPU") and
        # CUDA < 12.8, so enforce both BEFORE spending time on a build.
        if [[ "$SA_MAJOR" == "3" ]]; then
            $SA3_SUPPORTED || die "SageAttention 3 supports Blackwell GPUs only (compute capability 10.0 / 12.0 / 12.1). Detected: ${CC_RAW:-unknown}. Install SageAttention 2 instead."
            if ! pyrun -c "
import re
def vt(s): return tuple(int(x) for x in re.findall(r'\d+', s))
exit(0 if vt('$NVCC_VER') >= vt('12.8') else 1)
" 2>/dev/null; then
                error "SageAttention 3 requires a CUDA toolkit >= 12.8 (found nvcc $NVCC_VER)."
                printf '  Install a newer toolkit from\n'
                printf '  %shttps://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/%s\n' "$CYAN" "$NC"
                die  "Re-run this script after installing the toolkit."
            fi
            # README-stated floors that setup.py does not enforce — warn only.
            (( PY_MAJOR == 3 && PY_MINOR < 13 )) && \
                warn "SageAttention 3's README asks for Python >= 3.13 (found $PYTHON_VERSION) — attempting anyway."
            TORCH_SA3_OK=$(pyrun -c "
import re
def vt(s): return tuple(int(x) for x in re.findall(r'\d+', s.split('+')[0]))
print('ok' if vt('$TORCH_VER') >= (2, 8, 0) else 'old')
" 2>/dev/null || echo "old")
            [[ "$TORCH_SA3_OK" == "ok" ]] || \
                warn "SageAttention 3's README asks for torch >= 2.8.0 (found $TORCH_VER) — attempting anyway."
            # SA3's install_requires lists einops; the build below uses --no-deps.
            if [[ "$(dist_version einops)" == "not_installed" ]]; then
                warn "einops (SageAttention 3 runtime dependency) is not installed."
                if confirm "  Install now?" y; then
                    pyrun -m pip install einops --quiet \
                        || warn "einops install failed — install it manually before using sageattn3."
                fi
            fi
        fi

        # ── 6b.1  Host compiler selection ─────────────────────────────────────
        # System gcc/g++ is the safe default: nvcc supports it on every CUDA
        # release Ubuntu 24.04 ships, and it is never sandboxed. We do NOT infer
        # the compiler from the Python build (python-build-standalone interpreters
        # are built with Clang — that says nothing about which host compiler to
        # use for extensions), and we never auto-pick a compiler that resolves
        # under /snap: snap confinement hides /mnt, /media and the host /tmp from
        # the compiler, which makes every #include under those paths fail with
        # "file not found" even though the files exist.
        compiler_path() {
            # Echo an absolute, non-snap path for a compiler name, or fail.
            local p
            p=$(command -v "$1" 2>/dev/null) || return 1
            [[ -x "$p" ]] || return 1
            case "$(readlink -f "$p" 2>/dev/null || printf '%s' "$p")" in
                /snap/*) return 1 ;;
            esac
            printf '%s\n' "$p"
        }

        ensure_compiler() {
            local gcc_bin gxx_bin clang_bin clangxx_bin
            if [[ -n "${CC:-}" && -n "${CXX:-}" ]]; then
                info "Using host compiler from environment: CC=$CC CXX=$CXX"
                return 0
            fi
            if gcc_bin=$(compiler_path gcc) && gxx_bin=$(compiler_path g++); then
                export CC="$gcc_bin" CXX="$gxx_bin"
                success "C++ compiler: $("$CXX" --version | head -1)"
                return 0
            fi
            warn "No system gcc/g++ found — a C++ compiler is required for the source build."
            if confirm "  Install 'build-essential' via apt now?" y; then
                apt_install build-essential || true
                if gcc_bin=$(compiler_path gcc) && gxx_bin=$(compiler_path g++); then
                    export CC="$gcc_bin" CXX="$gxx_bin"
                    success "C++ compiler: $("$CXX" --version | head -1)"
                    return 0
                fi
            fi
            if clangxx_bin=$(compiler_path clang++) && clang_bin=$(compiler_path clang); then
                export CC="$clang_bin" CXX="$clangxx_bin"
                warn "gcc/g++ unavailable — falling back to clang: $("$CXX" --version | head -1)"
                return 0
            fi
            if confirm "  Install 'clang' via apt as a fallback compiler?" n; then
                apt_install clang || true
                if clangxx_bin=$(compiler_path clang++) && clang_bin=$(compiler_path clang); then
                    export CC="$clang_bin" CXX="$clangxx_bin"
                    success "C++ compiler: $("$CXX" --version | head -1)"
                    return 0
                fi
            fi
            die "No usable C++ compiler found. Run: sudo apt install build-essential"
        }

        # ── 6b.2  Python build dependencies ───────────────────────────────────
        ensure_build_deps() {
            info "Ensuring Python build deps (ninja, setuptools, wheel, packaging, pybind11)..."
            if pyrun -m pip install ninja setuptools wheel packaging pybind11 --quiet; then
                success "Python build dependencies present."
            else
                warn "pip install of build dependencies failed — the build below may fail."
            fi
            local venv_bin
            venv_bin="$(dirname "$PYTHON_BIN")"
            export PATH="${venv_bin}:${PATH}"
            if command -v ninja &>/dev/null; then
                success "ninja: $(ninja --version)"
            else
                warn "ninja not on PATH — the build falls back to the slower generator."
            fi
        }

        # ── 6b.3  torch dev headers ───────────────────────────────────────────
        # torch imports fine at runtime even when its C++ headers are missing
        # (stripped "portable" bundles, or venvs copied between machines whose
        # installs are dangling symlinks). A source build needs the headers.
        ensure_torch_headers() {
            TORCH_INCLUDE_DIR=$(pyrun -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'include'))" 2>/dev/null) \
                || die "Could not import torch to locate its headers — the PyTorch install in this environment is broken."
            PY_INCLUDE_DIR=$(pyrun -c "import sysconfig; print(sysconfig.get_paths()['include'])" 2>/dev/null) \
                || die "Could not query Python include path via sysconfig."

            # torch wheels vendor pybind11 inside torch/include, so both must exist.
            local -a missing_headers=()
            [[ -f "$TORCH_INCLUDE_DIR/torch/extension.h" ]] || missing_headers+=(torch/extension.h)
            [[ -f "$TORCH_INCLUDE_DIR/pybind11/pybind11.h" ]] || missing_headers+=(pybind11/pybind11.h)
            if (( ${#missing_headers[@]} == 0 )); then
                success "torch dev headers: $TORCH_INCLUDE_DIR"
                return 0
            fi

            error "PyTorch development headers are missing from this environment:"
            local h
            for h in "${missing_headers[@]}"; do
                printf '    %s/%s\n' "$TORCH_INCLUDE_DIR" "$h"
            done
            printf '\n'
            printf '  torch runs fine but ships no C++ headers. This happens with\n'
            printf '  stripped-down "portable" torch bundles, or with environments copied\n'
            printf '  between machines/drives whose files are dangling symlinks. SageAttention 2\n'
            printf '  cannot be compiled against such an install.\n\n'
            printf '  Fix: reinstall the SAME torch version from PyTorch'"'"'s wheel index so real\n'
            printf '  files land in site-packages:\n'
            printf '    %spip install --force-reinstall --no-deps torch==%s \\\n' "$CYAN" "${TORCH_VER%%+*}"
            printf '        --index-url https://download.pytorch.org/whl/cu%s%s\n\n' \
                "$(echo "$TORCH_CUDA_FOR_BUILD" | tr -d '.')" "$NC"
            if confirm "  Reinstall torch ${TORCH_VER%%+*} (CUDA ${TORCH_CUDA_FOR_BUILD}) into this environment now?" n; then
                if pyrun -m pip install --force-reinstall --no-deps "torch==${TORCH_VER%%+*}" \
                        --index-url "https://download.pytorch.org/whl/cu$(echo "$TORCH_CUDA_FOR_BUILD" | tr -d '.')"; then
                    TORCH_INCLUDE_DIR=$(pyrun -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'include'))")
                    if [[ -f "$TORCH_INCLUDE_DIR/torch/extension.h" ]]; then
                        success "torch dev headers are now present."
                        return 0
                    fi
                    die "Headers still missing after reinstall — inspect $TORCH_INCLUDE_DIR manually."
                fi
                die "torch reinstall failed — run the pip command above manually, then re-run this script."
            fi
            die "Cannot build SageAttention 2 without torch dev headers."
        }

        # ── 6b.4  Toolchain probes ────────────────────────────────────────────
        # Compile a trivial TU against torch's headers with the exact include
        # dirs the real build will use, then a one-line nvcc probe. Catches —
        # BEFORE a long build — sandboxed compilers, broken toolchains and
        # nvcc/host-compiler mismatches.
        run_toolchain_probes() {
            local probe_dir
            probe_dir=$(mktemp -d -t sa2-probe.XXXXXX)

            printf '#include <torch/extension.h>\n#include <pybind11/pybind11.h>\nint main() { return 0; }\n' \
                > "$probe_dir/probe.cpp"

            if ! "$CXX" -std=c++17 -fsyntax-only \
                    -I"$TORCH_INCLUDE_DIR" \
                    -I"$TORCH_INCLUDE_DIR/torch/csrc/api/include" \
                    -I"$PY_INCLUDE_DIR" \
                    -I"${CUDA_HOME}/include" \
                    "$probe_dir/probe.cpp" > "$probe_dir/cxx.log" 2>&1; then
                warn "C++ toolchain probe FAILED."
                if grep -qE "file not found|No such file or directory" "$probe_dir/cxx.log"; then
                    # Re-check the headers on disk to distinguish "header really
                    # absent" from "compiler cannot see files that are there".
                    local -a present=()
                    [[ -f "$TORCH_INCLUDE_DIR/torch/extension.h" ]] && present+=(torch/extension.h)
                    [[ -f "$TORCH_INCLUDE_DIR/pybind11/pybind11.h" ]] && present+=(pybind11/pybind11.h)
                    if (( ${#present[@]} == 2 )); then
                        printf '  The headers exist on disk but the compiler cannot open them — the\n'
                        printf '  signature of a sandboxed compiler (snap builds cannot read /mnt,\n'
                        printf '  /media or the host /tmp). Compiler in use: %s\n' "$CXX"
                        if [[ "$CXX" != "/usr/bin/g++" ]] && [[ -x /usr/bin/g++ ]]; then
                            if confirm "  Switch the host compiler to /usr/bin/g++ and re-probe?" y; then
                                export CC=/usr/bin/gcc CXX=/usr/bin/g++
                                if "$CXX" -std=c++17 -fsyntax-only \
                                        -I"$TORCH_INCLUDE_DIR" \
                                        -I"$TORCH_INCLUDE_DIR/torch/csrc/api/include" \
                                        -I"$PY_INCLUDE_DIR" \
                                        -I"${CUDA_HOME}/include" \
                                        "$probe_dir/probe.cpp" > "$probe_dir/cxx2.log" 2>&1; then
                                    success "Probe passes with /usr/bin/g++."
                                    rm -rf "$probe_dir"
                                    return 0
                                fi
                                cat "$probe_dir/cxx2.log" >&2
                                die "Probe still failing with g++ — see the output above."
                            fi
                        fi
                    else
                        printf '  Headers really are absent from %s —\n' "$TORCH_INCLUDE_DIR"
                        printf '  the torch install is incomplete (see the message above the probe).\n'
                    fi
                fi
                cat "$probe_dir/cxx.log" >&2
                die "The C++ toolchain cannot compile against torch's headers — see the probe output above."
            fi
            success "C++ probe OK — torch/pybind11/Python headers are visible to $CXX."

            printf '__global__ void sa2_probe() {}\n' > "$probe_dir/probe.cu"
            if ! "${CUDA_HOME}/bin/nvcc" -ccbin "$CC" -c "$probe_dir/probe.cu" \
                    -o "$probe_dir/probe.o" > "$probe_dir/nvcc.log" 2>&1; then
                warn "nvcc probe FAILED."
                if grep -qiE "unsupported (GNU|clang) version|versions later than" "$probe_dir/nvcc.log"; then
                    error "nvcc rejects the host compiler ($CC)."
                    if confirm "  Install gcc-12/g++-12 (a safe CUDA host compiler) and retry?" y; then
                        if apt_install gcc-12 g++-12; then
                            export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12
                            if "${CUDA_HOME}/bin/nvcc" -ccbin "$CC" -c "$probe_dir/probe.cu" \
                                    -o "$probe_dir/probe.o" > "$probe_dir/nvcc2.log" 2>&1; then
                                success "nvcc probe OK with gcc-12."
                                rm -rf "$probe_dir"
                                return 0
                            fi
                            cat "$probe_dir/nvcc2.log" >&2
                            die "nvcc probe still failing — see the output above."
                        fi
                    fi
                fi
                cat "$probe_dir/nvcc.log" >&2
                die "nvcc could not compile a trivial kernel with $CC — see the probe output above."
            fi
            success "nvcc probe OK — nvcc $NVCC_VER accepts $CC as host compiler."
            rm -rf "$probe_dir"
        }

        ensure_compiler
        ensure_build_deps
        ensure_torch_headers
        run_toolchain_probes

        # ── 6b.5  Clone / update the SageAttention repo ───────────────────────
        SA_DIR_DEFAULT="$(dirname "$COMFYUI_DIR")/SageAttention"
        SA_DIR="$SA_DIR_DEFAULT"
        # Space-in-path handling: we control the clone dir, so if the *default*
        # location would contain a space (because ComfyUI's parent does), relocate
        # into a mktemp dir instead. We do NOT attempt to patch spaced venv paths.
        if [[ "$SA_DIR" == *" "* ]]; then
            SA_DIR=$(mktemp -d -t sageattention-src.XXXXXX)
            warn "ComfyUI parent path contains spaces; cloning into $SA_DIR instead of $SA_DIR_DEFAULT."
            warn "Spaced venv/Python paths are not supported for source builds."
        fi
        if [[ -d "$SA_DIR/.git" ]]; then
            info "Updating SageAttention clone at $SA_DIR ..."
            if ! git -C "$SA_DIR" pull --ff-only; then
                warn "git pull --ff-only failed; resetting to origin/HEAD."
                git -C "$SA_DIR" fetch origin
                git -C "$SA_DIR" reset --hard origin/HEAD
            fi
        else
            info "Cloning SageAttention into $SA_DIR ..."
            git clone https://github.com/thu-ml/SageAttention.git "$SA_DIR"
        fi

        # ── 6b.6  Parallel jobs ───────────────────────────────────────────────
        if [[ -n "$MAX_JOBS_ARG" ]]; then
            MAX_JOBS="$MAX_JOBS_ARG"
        else
            MAX_JOBS="$(nproc 2>/dev/null || echo 4)"
        fi
        if ! $NON_INTERACTIVE; then
            ask "  Parallel compile jobs? [default: $MAX_JOBS]: " MAX_JOBS_INPUT
            MAX_JOBS="${MAX_JOBS_INPUT:-$MAX_JOBS}"
        fi
        [[ "$MAX_JOBS" =~ ^[0-9]+$ ]] || MAX_JOBS=4
        info "MAX_JOBS=$MAX_JOBS"

        # ── 6b.7  Build (regular install — NOT editable, so the installed
        #    package keeps working even if the clone dir is later deleted) ────
        BUILD_LOG="$PWD/sageattention_build.log"
        if ! touch "$BUILD_LOG" 2>/dev/null; then
            BUILD_LOG=$(mktemp -t sageattention_build.XXXXXX.log)
        fi
        : > "$BUILD_LOG"

        export MAX_JOBS CUDA_HOME="$CUDA_HOME_FOUND"

        # v2 honours the README's speed flags. v3's setup.py tunes nvcc itself
        # (--threads 4) and lives in its own subdirectory of the same clone;
        # its build also shallow-clones CUTLASS, which needs internet access.
        SA_BUILD_DIR="$SA_DIR"
        declare -a SA_BUILD_EXTRA_ENV=()
        if [[ "$SA_MAJOR" == "3" ]]; then
            SA_BUILD_DIR="$SA_DIR/sageattention3_blackwell"
            [[ -d "$SA_BUILD_DIR" ]] || \
                die "'$SA_BUILD_DIR' not found — the SageAttention repo layout changed?"
            info "The SageAttention 3 build also shallow-clones CUTLASS (needs internet)."
        else
            SA_BUILD_EXTRA_ENV+=(EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8")
        fi

        info "Building (MAX_JOBS=$MAX_JOBS). Log: $BUILD_LOG"
        cd "$SA_BUILD_DIR"

        # ── 6b.8  Build-failure diagnosis ─────────────────────────────────────
        # Scans the build log for well-known missing-dependency patterns,
        # explains them, and offers to install what is missing. Sets
        # DIAGNOSIS_AUTOFIX=true when a fix was applied automatically.
        DIAGNOSIS_AUTOFIX=false
        diagnose_build_failure() {
            local log="$1"
            DIAGNOSIS_AUTOFIX=false
            header "Build failure diagnosis"

            if grep -qE "Python\.h" "$log" && grep -qE "No such file or directory|file not found" "$log"; then
                error "Missing Python development headers (Python.h)."
                printf '  If this is the system Python, install them with:\n'
                printf '    %ssudo apt install python3-dev%s\n' "$CYAN" "$NC"
                printf '  Bundled/standalone Pythons normally ship their own headers —\n'
                printf '  check the include path reported by the probe step.\n'
            elif grep -qE "file not found|No such file or directory" "$log" && \
                 grep -qE "torch/(extension|version)\.h|ATen/[A-Za-z/]+\.h|pybind11/pybind11\.h|cuda_runtime\.h" "$log"; then
                error "The compiler could not find torch/pybind11/CUDA headers."
                printf '  Either PyTorch ships no dev headers (stripped "portable" bundles do\n'
                printf '  this) or the compiler is sandboxed (snap clang cannot read /mnt, /media\n'
                printf '  or host /tmp). The toolchain probe in Step 6b catches both — check its\n'
                printf '  output and re-run.\n'
            elif grep -qi "command not found" "$log" && ! command -v ninja &>/dev/null; then
                warn "ninja is missing from the environment."
                if confirm "  Install ninja into the Python environment now?" y; then
                    if pyrun -m pip install ninja --quiet; then
                        export PATH="$(dirname "$PYTHON_BIN"):${PATH}"
                        success "ninja installed: $(ninja --version 2>/dev/null || echo '?')"
                        DIAGNOSIS_AUTOFIX=true
                    else
                        warn "ninja install failed."
                    fi
                fi
            elif grep -qiE "unsupported (GNU|clang) version|gcc versions later than" "$log"; then
                error "nvcc rejects the host compiler version."
                if confirm "  Install gcc-12/g++-12 (a safe CUDA host compiler) and use them?" y; then
                    if apt_install gcc-12 g++-12; then
                        export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12
                        success "Host compiler switched to gcc-12."
                        DIAGNOSIS_AUTOFIX=true
                    fi
                fi
            elif grep -qi "unsupported" "$log" && grep -qiE "arch|compute_" "$log"; then
                error "The CUDA toolkit does not support this GPU's architecture."
                printf '  Your nvcc ($NVCC_VER) is too old for the detected compute capability\n'
                printf '  (%s). Install a newer toolkit from NVIDIA matching PyTorch (%s):\n' "$CC_RAW" "$TORCH_CUDA_FOR_BUILD"
                printf '    %shttps://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/%s\n' "$CYAN" "$NC"
            else
                warn "No specific diagnosis — first failure context:"
                grep -B2 -A8 -m1 -E "FAILED|error:" "$log" | head -30 >&2 || true
            fi
            printf '  Full build log: %s%s%s\n' "$CYAN" "$log" "$NC"
        }

        # Build + diagnose + retry loop (max 3 attempts). Relax errexit/pipefail
        # around the pip call so a failure can be reported and remedied instead
        # of aborting the script; the real exit code comes from PIPESTATUS[0].
        run_build() {
            if [[ "$ENV_TYPE" == "uv" ]]; then
                "$UV_BIN" run --project "$COMFYUI_DIR" \
                    env MAX_JOBS="$MAX_JOBS" CUDA_HOME="$CUDA_HOME" \
                    CC="${CC}" CXX="${CXX}" \
                    ${SA_BUILD_EXTRA_ENV[@]+"${SA_BUILD_EXTRA_ENV[@]}"} \
                    python -m pip install . --no-build-isolation --no-deps
            else
                env MAX_JOBS="$MAX_JOBS" CUDA_HOME="$CUDA_HOME" \
                    CC="${CC}" CXX="${CXX}" \
                    ${SA_BUILD_EXTRA_ENV[@]+"${SA_BUILD_EXTRA_ENV[@]}"} \
                    "$PYTHON_BIN" -m pip install . --no-build-isolation --no-deps
            fi
        }

        BUILD_ATTEMPT=0
        while true; do
            BUILD_ATTEMPT=$((BUILD_ATTEMPT + 1))
            set +e
            set +o pipefail
            run_build 2>&1 | tee "$BUILD_LOG"
            BUILD_EXIT=${PIPESTATUS[0]}
            set -o pipefail
            set -e
            [[ "${BUILD_EXIT:-0}" -eq 0 ]] && break

            diagnose_build_failure "$BUILD_LOG"
            if (( BUILD_ATTEMPT >= 3 )); then
                die "Build still failing after $BUILD_ATTEMPT attempts. Log: $BUILD_LOG"
            fi
            if $NON_INTERACTIVE && ! $DIAGNOSIS_AUTOFIX; then
                die "Build failed (non-interactive; no automatic fix was available). Log: $BUILD_LOG"
            fi
            if ! $NON_INTERACTIVE && ! confirm "  Retry the build?" y; then
                die "Aborting after failed build. Log: $BUILD_LOG"
            fi
        done

        SA_VER_NEW=$(dist_version "$SA_DIST_NAME")
        [[ "$SA_VER_NEW" != "not_installed" ]] || {
            error "$SA_DIST_NAME could not be imported after build."
            die "Log: $BUILD_LOG"
        }
        success "$SA_DIST_NAME $SA_VER_NEW (SageAttention $SA_MAJOR) installed from source."
    fi
fi

# ── 7. Runtime smoke test ─────────────────────────────────────────────────────
# `import sageattention` succeeding does NOT mean the kernel runs. A tiny
# attention call catches the common "compiled but crashes at runtime" failure.
header "Step 7 of 8 – Runtime smoke test"
if [[ "$SA_MAJOR" == "3" ]]; then
    # sageattn3 is a separate package; use FP4-friendly shapes (head_dim 128).
    SMOKE=$(pyrun - <<'EOF' 2>&1 || true
import torch
from sageattn3 import sageattn3_blackwell
q = torch.randn(1, 16, 256, 128, dtype=torch.float16, device="cuda")
k = torch.randn(1, 16, 256, 128, dtype=torch.float16, device="cuda")
v = torch.randn(1, 16, 256, 128, dtype=torch.float16, device="cuda")
out = sageattn3_blackwell(q, k, v)
assert out.shape == q.shape
print("SMOKE_OK")
EOF
)
else
    SMOKE=$(pyrun - <<'EOF' 2>&1 || true
import torch
from sageattention import sageattn
q = torch.randn(1, 8, 16, 64, dtype=torch.float16, device="cuda")
k = torch.randn(1, 8, 16, 64, dtype=torch.float16, device="cuda")
v = torch.randn(1, 8, 16, 64, dtype=torch.float16, device="cuda")
out = sageattn(q, k, v)
assert out.shape == q.shape
print("SMOKE_OK")
EOF
)
fi
if echo "$SMOKE" | grep -q "SMOKE_OK"; then
    success "sageattn(q,k,v) ran on CUDA — runtime OK."
else
    warn "Runtime smoke test did not complete successfully:"
    printf "  %s\n" "$SMOKE" | head -20 >&2
    warn "SageAttention imported but a CUDA kernel call failed. See output above."
    if ! $NON_INTERACTIVE; then
        confirm "  Continue anyway?" y || die "Aborting."
    fi
fi

# ── 8. Summary ────────────────────────────────────────────────────────────────
header "Step 8 of 8 – Done!"

printf '\n'
printf '  %sInstallation summary%s\n' "$BOLD" "$NC"
printf '  ┌──────────────────────────────────────────────────────┐\n'
printf '  │  Environment   : %s%s%s\n' "$CYAN" "$ENV_TYPE" "$NC"
printf '  │  Python        : %s%s%s\n' "$CYAN" "$PYTHON_VERSION" "$NC"
printf '  │  PyTorch       : %s%s%s (CUDA %s)\n' "$CYAN" "$TORCH_VER" "$NC" "$TORCH_CUDA_FOR_BUILD"
printf '  │  Triton        : %s%s%s\n' "$CYAN" "$(dist_version triton)" "$NC"
printf '  │  SageAttention : %s%s%s  (v%s, %s)\n' "$CYAN" "$(dist_version "$SA_DIST_NAME")" "$NC" "$SA_MAJOR" "$SA_METHOD"
if [[ "$SA_MAJOR" == "3" ]]; then
    SA2_ALSO=$(dist_version sageattention)
    [[ "$SA2_ALSO" != "not_installed" ]] && \
        printf '  │  sageattention: %s%s%s  (v1/v2, coinstalled)\n' "$CYAN" "$SA2_ALSO" "$NC"
fi
printf '  └──────────────────────────────────────────────────────┘\n'
printf '\n'
if [[ "$SA_MAJOR" == "3" ]]; then
    printf '  %sUsing SageAttention 3 in ComfyUI:%s the core --use-sage-attention\n' "$BOLD" "$NC"
    printf '  flag targets SageAttention 1/2. SageAttention 3 is selected per-model\n'
    printf '  via the KJNodes %sPatch Sage Attention KJ%s node (sageattn3_fp4_cuda mode)\n' "$CYAN" "$NC"
    printf '  or a dedicated SageAttention3 custom node.\n\n'
else
    printf '  %sLaunch ComfyUI with SageAttention:%s\n\n' "$BOLD" "$NC"
    if [[ "$ENV_TYPE" == "uv" ]]; then
        printf '  %scd %s%s\n' "$CYAN" "$COMFYUI_DIR" "$NC"
        printf '  %suv run python main.py --use-sage-attention%s\n' "$CYAN" "$NC"
    else
        printf '  %scd %s%s\n' "$CYAN" "$COMFYUI_DIR" "$NC"
        printf '  %s%s main.py --use-sage-attention%s\n' "$CYAN" "$PYTHON_BIN" "$NC"
    fi
    printf '\n'
    printf '  %sNote:%s If using the ComfyUI Desktop app, enable SageAttention\n' "$YELLOW" "$NC"
    printf '  from the ComfyUI settings UI instead.\n\n'
fi
success "All done. Enjoy faster inference!"
