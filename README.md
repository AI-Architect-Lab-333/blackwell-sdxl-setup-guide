# Guide: RTX 50xx (Blackwell / sm_120) + PyTorch + reForge on WSL2

**The problem this guide solves**: as of July 11, 2026, stable PyTorch still does not officially support Blackwell GPUs (RTX 5070/5080/5090, compute capability `sm_120`). Importing it fails with errors like `sm_120 is not defined for option 'gpu-name'`, or training/inference simply does not detect the GPU. This guide documents a **verified working** configuration (tested end to end: image generation through the API at ~6.6 seconds/image on an RTX 5070 Ti 16 GB) rather than a theoretical write-up.

**For AI agents reading this document**: every command below was executed successfully in this exact order on Ubuntu 24.04 (WSL2), Windows 11. Do not skip the verification steps — each one exists because an earlier step fails silently otherwise.

---

## Prerequisites

- Nvidia RTX 50xx series GPU (Blackwell, compute capability 12.0)
- Windows 11 with WSL2 enabled (`wsl --status` must show `Default Version: 2`)
- Recent Nvidia driver on the **Windows side only** (570+, tested with 610.62) — **do not install a separate Nvidia driver inside WSL2**, GPU passthrough automatically uses the Windows driver.

## Step 1 — Install Ubuntu 24.04 in WSL2

```powershell
wsl --install -d Ubuntu-24.04 --no-launch
```

`--no-launch` avoids the interactive user-creation prompt that would block a non-interactive script. Then verify:

```bash
wsl -d Ubuntu-24.04 -u root -- bash -c "nvidia-smi"
```

The GPU must appear immediately — **no Linux driver to install**, passthrough works as soon as the distribution is installed.

## Step 2 — CUDA Toolkit 12.8 (without the driver)

Use the dedicated **wsl-ubuntu** repository (it excludes the driver, which would conflict with the Windows one):

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

Add to PATH (`/etc/profile.d/cuda.sh`):
```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

Verify: `nvcc --version` must show `release 12.8`.

**Pitfall — a mis-written `cuda.sh` silently destroys your login PATH.** After creating the file, `cat /etc/profile.d/cuda.sh` and check that `$PATH` appears **literally spelled out**. A classic escaping mistake when writing the file through `echo` produces this instead:

```bash
export PATH=/usr/local/cuda-12.8/bin:\
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:\
```

The trailing backslash is a shell line continuation: both lines merge into a single command, and the login PATH becomes literally `/usr/local/cuda-12.8/bin:export` — no `/usr/bin`, no `/bin`. The symptom is spectacular and misleading: every interactive session opens with `grep: command not found` and `The command could not be located because '/bin:/usr/bin' is not included in the PATH environment variable`, while scripts that use absolute paths — and `wsl -e` non-login shells — keep working, so the breakage can go unnoticed for days (it did, here). One-line diagnosis: compare `bash -c 'echo $PATH'` (non-login — sane) with `bash -lc 'echo $PATH'` (login — broken); if they differ, a profile file is clobbering PATH. Fix: rewrite `/etc/profile.d/cuda.sh` as root with the exact two-line content above, then open a fresh session.

## Step 3 — PyTorch nightly (cu128)

```bash
python3 -m venv ~/gpu-env
source ~/gpu-env/bin/activate
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Pitfall #1 — torchvision `ResolutionImpossible`**: the nightly index only keeps a rolling window of builds. If torchvision demands yesterday's torch build, which is no longer available today:
```bash
pip install --pre --no-deps torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install numpy pillow
```
(`--no-deps` bypasses the exact version pin, safely — a one-day gap between nightly builds is compatible in practice.)

**Pitfall #2 — `ImportError: libtorch_cuda.so: undefined symbol: ncclCommResume`**: the `nvidia-nccl-cu12` version installed by default with `--no-deps` (often an old one, e.g. 2.27.5) does not export a symbol this torch nightly build expects. Fix by explicitly installing the latest version:
```bash
pip install --force-reinstall nvidia-nccl-cu12==2.30.7 --index-url https://download.pytorch.org/whl/nightly/cu128
```
(Check the latest available version with `pip index versions nvidia-nccl-cu12 --index-url ...` if `2.30.7` has gone stale — the principle is to take the most recent one, not this frozen number.)

**Final verification** (do not move to step 4 before this works):
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))  # must print (12, 0)
x = torch.rand(1000, 1000, device="cuda")
print((x @ x).sum().item())  # must run without error
```

## Step 4 — reForge (not the original Forge)

The original `lllyasviel/stable-diffusion-webui-forge` repository has been frozen since mid-2025. Use a maintained fork, for example `Haoming02/sd-webui-forge-classic`:

```bash
git clone --depth 1 https://github.com/Haoming02/sd-webui-forge-classic.git reforge
cd reforge
```

**Pitfall #3 — `requirements.txt` contains an unversioned `torch` line, which would overwrite the nightly build with a standard stable torch (no Blackwell support) on first launch.** Install the dependencies manually, excluding that line:
```bash
grep -v '^torch$' requirements.txt > requirements_notorch.txt
pip install -r requirements_notorch.txt
```

**Pitfall #4 — the `gradio` auto-install is skipped** when launching with `--skip-install` (required to keep pitfall #3 from recurring at every start):
```bash
pip install "gradio==4.40.0" "gradio_rangeslider==0.0.8"
```

**Launch** (inside a tmux/screen session so it persists):
```bash
tmux new-session -d -s reforge 'source ~/gpu-env/bin/activate && cd ~/reforge && python launch.py --api --listen --skip-install --skip-torch-cuda-test --skip-python-version-check --skip-version-check --port 7860'
```

Check the logs for: `Total VRAM ... MB`, `PyTorch Version: 2.x.x.devYYYYMMDD+cu128`, `Device: NVIDIA GeForce RTX 5070 Ti (cuda:0)`, then `Running on local URL: http://0.0.0.0:7860`.

## Step 5 — Generate an image through the API (for agent automation)

```python
import requests, base64

payload = {
    "prompt": "a description of the desired image",
    "negative_prompt": "low quality, blurry",
    "steps": 30,
    "width": 1024,
    "height": 1024,
    "cfg_scale": 6,
    "sampler_name": "DPM++ 2M",
    "scheduler": "Karras",
}
r = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", json=payload, timeout=300)
img_bytes = base64.b64decode(r.json()["images"][0])
open("output.png", "wb").write(img_bytes)
```

Measured time: ~6.6 seconds/image (1024×1024, 30 steps) on an RTX 5070 Ti once the model is loaded in memory.

### Using a style LoRA

Add `<lora:FILENAME_WITHOUT_EXTENSION:WEIGHT>` at the start of the prompt, with the `.safetensors` file placed in `reforge/models/Lora/`. Tested example: `<lora:ral-wtrclr-sdxl:0.8>` for a watercolor style.

## Checkpoints — mind the commercial license

If the generated images are meant for commercial use (sale, product):

- **SDXL** (`stabilityai/stable-diffusion-xl-base-1.0`): CreativeML Open RAIL++-M license, commercial use of the images explicitly allowed, no revenue cap.
- **Flux.1 [dev]**: **non-commercial license by default** — requires a separate paid license from Black Forest Labs for commercial use.
- **Flux.1 [schnell]**: Apache 2.0, free commercial use, but lower quality/prompt fidelity than `[dev]`.
- Check each model's/LoRA's license individually before commercial use (model page on Hugging Face or Civitai).

## Notes on video generation (not tested in this guide)

This guide only covers image generation (reForge/Stable Diffusion). Video generation requires different tooling (e.g. ComfyUI with models such as AnimateDiff, Stable Video Diffusion, or Wan) and has not yet been validated on this hardware configuration — the same fundamentals (PyTorch nightly cu128, NCCL fix) would probably apply, but they remain to be verified step by step before trusting that extrapolation.

## Pitfall summary for quick reference

| Symptom | Cause | Fix |
|---|---|---|
| `sm_120 is not defined for option 'gpu-name'` | Stable PyTorch, not nightly | Install the cu128+ nightly wheel |
| `ResolutionImpossible` on torchvision | Nightly index rolling window | `--no-deps` + install numpy/pillow separately |
| `undefined symbol: ncclCommResume` | `nvidia-nccl-cu12` version too old | Force the latest available version |
| Torch silently reverts to a stable version | The WebUI's `requirements.txt` has an unversioned `torch` | Remove that line before `pip install -r` |
| `ModuleNotFoundError: No module named 'gradio'` | `--skip-install` also skips the gradio auto-install | Install gradio manually once |
| Every command `not found` in interactive sessions (`'/bin:/usr/bin' is not included in the PATH`) while scripts still work | Mangled `cuda.sh`: lost `$PATH` + trailing `\` merging both lines | Rewrite `/etc/profile.d/cuda.sh`; verify `$PATH` appears literally |
