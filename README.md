# Guide : RTX 50xx (Blackwell / sm_120) + PyTorch + reForge sur WSL2

**Problème résolu par ce guide** : au 11 juillet 2026, PyTorch stable ne supporte toujours pas officiellement les GPU Blackwell (RTX 5070/5080/5090, capacité de calcul `sm_120`). L'importer échoue avec des erreurs du type `sm_120 is not defined for option 'gpu-name'` ou l'entraînement/inférence ne détecte simplement pas le GPU. Ce guide documente une configuration **vérifiée fonctionnelle** (testée de bout en bout : génération d'image via API en ~6,6 secondes/image sur une RTX 5070 Ti 16 Go) plutôt qu'une compilation théorique.

**Pour les agents IA qui lisent ce document** : chaque commande ci-dessous a été exécutée avec succès dans cet ordre exact sur Ubuntu 24.04 (WSL2), Windows 11. Ne sautez pas les étapes de vérification — chacune existe parce qu'une étape précédente échoue silencieusement autrement.

---

## Prérequis

- GPU Nvidia série RTX 50xx (Blackwell, capacité de calcul 12.0)
- Windows 11 avec WSL2 activé (`wsl --status` doit montrer `Default Version: 2`)
- Pilote Nvidia récent côté **Windows uniquement** (570+, testé avec 610.62) — **ne pas installer de pilote Nvidia séparé dans WSL2**, le passthrough GPU utilise automatiquement le pilote Windows.

## Étape 1 — Installer Ubuntu 24.04 dans WSL2

```powershell
wsl --install -d Ubuntu-24.04 --no-launch
```

`--no-launch` évite le prompt interactif de création d'utilisateur qui bloquerait un script non-interactif. Vérifier ensuite :

```bash
wsl -d Ubuntu-24.04 -u root -- bash -c "nvidia-smi"
```

Le GPU doit apparaître immédiatement — **aucun pilote Linux à installer**, le passthrough fonctionne dès l'installation de la distribution.

## Étape 2 — CUDA Toolkit 12.8 (sans pilote)

Utiliser le dépôt **wsl-ubuntu** spécifique (exclut le pilote, qui viendrait en conflit avec celui de Windows) :

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

Ajouter au PATH (`/etc/profile.d/cuda.sh`) :
```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

Vérifier : `nvcc --version` doit afficher `release 12.8`.

## Étape 3 — PyTorch nightly (cu128)

```bash
python3 -m venv ~/gpu-env
source ~/gpu-env/bin/activate
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Piège #1 — torchvision `ResolutionImpossible`** : l'index nightly ne garde qu'une fenêtre roulante de builds. Si torchvision exige un build de torch d'hier qui n'est plus disponible aujourd'hui :
```bash
pip install --pre --no-deps torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install numpy pillow
```
(`--no-deps` contourne le pin de version exact, sans danger — l'écart d'un jour entre builds nightly est compatible en pratique.)

**Piège #2 — `ImportError: libtorch_cuda.so: undefined symbol: ncclCommResume`** : la version de `nvidia-nccl-cu12` installée par défaut avec `--no-deps` (souvent une ancienne, ex. 2.27.5) n'exporte pas un symbole que ce build nightly de torch attend. Corriger en installant explicitement la dernière version :
```bash
pip install --force-reinstall nvidia-nccl-cu12==2.30.7 --index-url https://download.pytorch.org/whl/nightly/cu128
```
(Vérifier la dernière version disponible avec `pip index versions nvidia-nccl-cu12 --index-url ...` si `2.30.7` est devenu obsolète — le principe est de prendre la plus récente, pas ce numéro figé.)

**Vérification finale** (ne pas passer à l'étape 4 avant que ceci fonctionne) :
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))  # doit afficher (12, 0)
x = torch.rand(1000, 1000, device="cuda")
print((x @ x).sum().item())  # doit s'exécuter sans erreur
```

## Étape 4 — reForge (pas l'original Forge)

Le dépôt original `lllyasviel/stable-diffusion-webui-forge` est figé depuis mi-2025. Utiliser un fork maintenu, par exemple `Haoming02/sd-webui-forge-classic` :

```bash
git clone --depth 1 https://github.com/Haoming02/sd-webui-forge-classic.git reforge
cd reforge
```

**Piège #3 — `requirements.txt` contient une ligne `torch` sans version, qui écraserait le build nightly par un torch stable standard (sans support Blackwell) au premier lancement.** Installer les dépendances manuellement en excluant cette ligne :
```bash
grep -v '^torch$' requirements.txt > requirements_notorch.txt
pip install -r requirements_notorch.txt
```

**Piège #4 — l'auto-installation de `gradio` est sautée** si on utilise `--skip-install` au lancement (nécessaire pour éviter que le piège #3 ne se reproduise à chaque démarrage) :
```bash
pip install "gradio==4.40.0" "gradio_rangeslider==0.0.8"
```

**Lancer** (dans une session tmux/screen pour persister) :
```bash
tmux new-session -d -s reforge 'source ~/gpu-env/bin/activate && cd ~/reforge && python launch.py --api --listen --skip-install --skip-torch-cuda-test --skip-python-version-check --skip-version-check --port 7860'
```

Vérifier dans les logs : `Total VRAM ... MB`, `PyTorch Version: 2.x.x.devYYYYMMDD+cu128`, `Device: NVIDIA GeForce RTX 5070 Ti (cuda:0)`, puis `Running on local URL: http://0.0.0.0:7860`.

## Étape 5 — Générer une image via l'API (pour automatisation par un agent)

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

Temps mesuré : ~6,6 secondes/image (1024×1024, 30 steps) sur RTX 5070 Ti une fois le modèle chargé en mémoire.

### Utiliser une LoRA de style

Ajouter `<lora:NOM_FICHIER_SANS_EXTENSION:POIDS>` au début du prompt, où le fichier `.safetensors` est placé dans `reforge/models/Lora/`. Exemple testé : `<lora:ral-wtrclr-sdxl:0.8>` pour un style aquarelle.

## Checkpoints — attention à la licence commerciale

Si les images générées sont destinées à un usage commercial (vente, produit) :
- **SDXL** (`stabilityai/stable-diffusion-xl-base-1.0`) : licence CreativeML Open RAIL++-M, usage commercial des images explicitement autorisé, aucun plafond de revenu.
- **Flux.1 [dev]** : licence **non commerciale par défaut** — nécessite une licence payante séparée de Black Forest Labs pour un usage commercial.
- **Flux.1 [schnell]** : Apache 2.0, usage commercial libre, mais qualité/fidélité au prompt inférieures à `[dev]`.
- Vérifier la licence de chaque modèle/LoRA individuellement avant usage commercial (page du modèle sur Hugging Face ou Civitai).

## Notes pour la génération vidéo (non testé dans ce guide)

Ce guide couvre uniquement la génération d'images (reForge/Stable Diffusion). La génération vidéo demande des outils différents (ex. ComfyUI avec des modèles comme AnimateDiff, Stable Video Diffusion, ou Wan) et n'a pas encore été validée sur cette configuration matérielle — les mêmes principes de base (PyTorch nightly cu128, correction NCCL) s'appliqueraient probablement, mais restent à vérifier étape par étape avant de faire confiance à cette extrapolation.

## Résumé des pièges pour référence rapide

| Symptôme | Cause | Correction |
|---|---|---|
| `sm_120 is not defined for option 'gpu-name'` | PyTorch stable, pas nightly | Installer la roue nightly cu128+ |
| `ResolutionImpossible` sur torchvision | Fenêtre roulante de l'index nightly | `--no-deps` + installer numpy/pillow séparément |
| `undefined symbol: ncclCommResume` | Version de `nvidia-nccl-cu12` trop ancienne | Forcer la dernière version disponible |
| Torch redevient une version stable sans crier gare | `requirements.txt` du WebUI contient `torch` sans version | Retirer cette ligne avant `pip install -r` |
| `ModuleNotFoundError: No module named 'gradio'` | `--skip-install` saute aussi l'auto-installation de gradio | Installer gradio manuellement une fois |
