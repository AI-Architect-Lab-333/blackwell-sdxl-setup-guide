"""
Exemple minimal : generer une image via l'API reForge, une fois le serveur
lance avec --api (voir README.md, etape 4).

Usage:
    python generate_image.py "a cozy watercolor illustration of a cat"
"""
import sys
import base64
import time
import requests

BASE_URL = "http://127.0.0.1:7860"


def generate(prompt: str, negative_prompt: str = "low quality, blurry, deformed",
             steps: int = 30, width: int = 1024, height: int = 1024,
             seed: int = -1, output_path: str = "output.png") -> str:
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "width": width,
        "height": height,
        "cfg_scale": 6,
        "sampler_name": "DPM++ 2M",
        "scheduler": "Karras",
        "seed": seed,
    }
    t0 = time.time()
    r = requests.post(f"{BASE_URL}/sdapi/v1/txt2img", json=payload, timeout=300)
    r.raise_for_status()
    elapsed = round(time.time() - t0, 1)
    data = r.json()
    img_bytes = base64.b64decode(data["images"][0])
    with open(output_path, "wb") as f:
        f.write(img_bytes)
    print(f"OK: {output_path} ({elapsed}s)")
    return output_path


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "a cozy watercolor illustration"
    generate(prompt)
