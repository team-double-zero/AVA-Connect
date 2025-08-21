#!/usr/bin/env python3
# comfy_request.py
# 예)
#  python3 comfy_request.py \
#    --prompt "a cinematic portrait of a young woman, soft lighting, 35mm, high detail" \
#    --n-prompt "blurry, low quality, artifacts" \
#    --ckpt qwen_image_fp8_e4m3fn.safetensors \
#    --width 832 --height 1216 --steps 30 --cfg 5.0 --seed 42
#
# LoRA 예)
#  python3 comfy_request.py \
#    --prompt "anime style, cherry blossoms, dynamic lighting, ultra-detailed" \
#    --n-prompt "blurry, low quality, artifacts" \
#    --ckpt qwen_image_fp8_e4m3fn.safetensors \
#    --lora Qwen-Image-Lightning-8steps-V1.0.safetensors --lora-strength 1.0 --lora-clip 1.0 \
#    --steps 8 --cfg 1.0 --seed 123
#
# 참고: 결과 이미지는 VM의 ComfyUI output 폴더에 저장되고,
#       스크립트는 다운로드하지 않고 /view URL만 출력합니다.

import argparse
import json
import time
from urllib.parse import urlencode

import requests


def build_payload(args):
    # 기본 노드 그래프 구성
    nodes = {
        "ckpt": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": args.ckpt},
        }
    }

    model_ref = ["ckpt", 0]
    clip_ref = ["ckpt", 1]
    vae_ref = ["ckpt", 2]

    if args.lora:
        nodes["lora"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_ref,
                "clip": clip_ref,
                "lora_name": args.lora,
                "strength_model": float(args.lora_strength),
                "strength_clip": float(args.lora_clip),
            },
        }
        model_ref = ["lora", 0]
        clip_ref = ["lora", 1]

    nodes.update({
        "pos": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": clip_ref, "text": args.prompt},
        },
        "neg": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": clip_ref, "text": args.n_prompt},
        },
        "latent": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": args.width, "height": args.height, "batch_size": args.batch},
        },
        "k": {
            "class_type": "KSampler",
            "inputs": {
                "model": model_ref,
                "positive": ["pos", 0],
                "negative": ["neg", 0],
                "latent_image": ["latent", 0],
                "sampler_name": args.sampler,
                "scheduler": args.scheduler,
                "seed": int(args.seed) if args.seed is not None else 0,
                "steps": args.steps,
                "cfg": args.cfg,
                "denoise": args.denoise,
            },
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["k", 0], "vae": vae_ref},
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {"images": ["decode", 0], "filename_prefix": args.prefix},
        },
    })

    return {"prompt": nodes, "client_id": args.client_id}


def post_prompt(base, payload, timeout=10):
    r = requests.post(f"{base}/prompt", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("prompt_id")


def fetch_history(base, prompt_id, timeout=10):
    r = requests.get(f"{base}/history/{prompt_id}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def wait_for_save_images(base, prompt_id, poll=0.5, max_wait=300):
    t0 = time.time()
    last = None
    while True:
        try:
            hist = fetch_history(base, prompt_id)
            last = hist
            entry = hist.get(prompt_id, {})
            outputs = entry.get("outputs", {})
            if "save" in outputs and "images" in outputs["save"]:
                imgs = outputs["save"]["images"]
                if imgs:
                    return imgs  # [{filename, subfolder, type}, ...]
        except requests.RequestException:
            pass
        if time.time() - t0 > max_wait:
            raise TimeoutError(f"Timed out waiting for result (prompt_id={prompt_id})\nLast history: {json.dumps(last, indent=2)}")
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    # 연결 설정
    ap.add_argument("--scheme", default="http", choices=["http", "https"], help="접속 스킴 (SSH 터널이면 http)")
    ap.add_argument("--host", default="127.0.0.1", help="ComfyUI 호스트 (SSH 터널이면 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8188, help="ComfyUI 포트")
    ap.add_argument("--client-id", default="cli", help="ComfyUI client_id")
    # 생성 파라미터
    ap.add_argument("--ckpt", required=True, help="체크포인트 파일명 (ComfyUI/models/checkpoints/ 내)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n-prompt", dest="n_prompt", default="worst quality, low quality, blurry, artifacts")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="normal")
    ap.add_argument("--denoise", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--prefix", default="qwen_cli")
    # LoRA 옵션
    ap.add_argument("--lora", default=None, help="LoRA 파일명 (ComfyUI/models/loras/ 내)")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--lora-clip", type=float, default=1.0)
    args = ap.parse_args()

    base = f"{args.scheme}://{args.host}:{args.port}"

    payload = build_payload(args)
    prompt_id = post_prompt(base, payload)
    print(f"[INFO] prompt_id: {prompt_id}")

    images = wait_for_save_images(base, prompt_id)
    # 각 이미지에 대해 /view URL 출력 (다운로드 X)
    for i, img in enumerate(images, 1):
        # ComfyUI /view 파라미터: filename, subfolder, type (type=output)
        q = {
            "filename": img.get("filename", ""),
            "subfolder": img.get("subfolder", ""),
            "type": img.get("type", "output") or "output",
        }
        url = f"{base}/view?{urlencode(q)}"
        print(f"[RESULT {i}] {url}")


if __name__ == "__main__":
    main()