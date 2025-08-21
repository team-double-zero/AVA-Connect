import os
import json
import time
import re
import torch
from diffusers import QwenImagePipeline

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"The config attributes \{'pooled_projection_dim': 768\}.*QwenImageTransformer2DModel"
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_multiple_of_8(x: int) -> int:
    return max(8, (x // 8) * 8)

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', name).strip('_')

def main():
    cfg_path = os.environ.get("CONFIG", "example.json")
    cfg = load_config(cfg_path)

    prompt = cfg.get("prompt", "")
    negative_prompt = cfg.get("negative_prompt", " ")

    width = ensure_multiple_of_8(int(cfg.get("width", 1024)))
    height = ensure_multiple_of_8(int(cfg.get("height", 1024)))
    steps = int(cfg.get("num_inference_steps", 30))

    # Qwen-Image는 guidance_scale 대신 true_cfg_scale을 사용
    true_cfg_scale = float(cfg.get("true_cfg_scale", cfg.get("guidance_scale", 3.5)))

    seed = cfg.get("seed", None)
    generator = torch.manual_seed(int(seed)) if seed is not None else None

    out_dir = cfg.get("out_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    filename = cfg.get("filename")
    if not filename:
        filename = f"generated_{int(time.time())}.png"
    filename = sanitize_filename(filename)
    out_path = os.path.join(out_dir, filename)

    # 모델 로드 (float16이 GPU 호환성이 더 넓음)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = QwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image",
        torch_dtype=dtype,
    ).to(device)

    img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        true_cfg_scale=true_cfg_scale,
        width=width,
        height=height,
        generator=generator,
    ).images[0]

    img.save(out_path)
    print(f"✅ Saved: {out_path}")
    print(f"- prompt: {prompt}")
    print(f"- size: {width}x{height}, steps: {steps}, true_cfg_scale: {true_cfg_scale}, seed: {seed}")

if __name__ == "__main__":
    main()
