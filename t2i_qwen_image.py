#!/usr/bin/env python3

import argparse, torch, os
from diffusers import Qwen2ImagePipeline  # 최신 diffusers에서는 Qwen2Image/QwenImage로 노출
from diffusers.utils import load_image
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="텍스트 프롬프트")
    ap.add_argument("--n_prompt", default="", help="네거티브 프롬프트")
    ap.add_argument("--out", default="qwen_out.png", help="출력 이미지 경로")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=3.5, help="이미지 가이던스(=CFG) 1.0~5.0 권장")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--repo", default="Qwen/Qwen-Image", help="HF 레포 (기본: Qwen/Qwen-Image)")
    ap.add_argument("--torch_dtype", default="bf16", choices=["bf16","fp16","fp32"])
    ap.add_argument("--lightning_lora", default="", help="Lightning LoRA safetensors 경로(선택)")
    ap.add_argument("--lora_scale", type=float, default=1.0, help="LoRA 스케일(0.5~1.0)")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.torch_dtype]

    # 1) 파이프라인 로드 (Qwen-Image 공식 파이프라인)
    #    * 최신 diffusers 필요 (docs/main 기준)
    pipe = Qwen2ImagePipeline.from_pretrained(
        args.repo,
        torch_dtype=dtype,
        variant=None
    ).to("cuda")

    # 2) (선택) Lightning LoRA 적용: 8step/4step 초고속 추론
    #    LoRA 파일이 있다면 적용 (예: models/loras/Qwen-Image-Lightning-8steps-V1.0.safetensors)
    if args.lightning_lora and os.path.isfile(args.lightning_lora):
        pipe.load_lora_weights(args.lightning_lora)
        pipe.fuse_lora(lora_scale=args.lora_scale)
        print(f"[INFO] Lightning LoRA 적용: {args.lightning_lora} (scale={args.lora_scale})")

    # 3) 생성
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.n_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        generator=generator,
    ).images[0]

    # 4) 저장
    out_path = os.path.abspath(args.out)
    image.save(out_path)
    print(f"[DONE] saved: {out_path}")

if __name__ == "__main__":
    main()