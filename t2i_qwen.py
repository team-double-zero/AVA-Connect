# t2i_qwen.py
import argparse
from pathlib import Path
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import make_image_grid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="로컬 Qwen 이미지 모델 경로 (diffusers 포맷)")
    ap.add_argument("--prompt", required=True, help="생성 프롬프트")
    ap.add_argument("--n_prompt", default="worst quality, low quality, blurry, artifacts", help="네거티브 프롬프트")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num", type=int, default=1, help="샘플 수")
    ap.add_argument("--lora", default=None, help="선택: LoRA 경로 또는 HF repo")
    ap.add_argument("--out", default="qwen_out.png", help="저장 파일명")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[INFO] loading model: {args.model_dir}")
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model_dir,
        torch_dtype=dtype
    )
    pipe = pipe.to(device)

    # 선택: LoRA 적용
    if args.lora:
        print(f"[INFO] loading LoRA: {args.lora}")
        pipe.load_lora_weights(args.lora)
        try:
            pipe.fuse_lora()  # 최신 diffusers에서는 성능 위해 fuse 가능
        except Exception:
            pass

    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    images = pipe(
        prompt=args.prompt,
        negative_prompt=args.n_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        num_images_per_prompt=args.num,
        generator=generator
    ).images

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(images) == 1:
        images[0].save(out_path)
        print(f"[OK] saved: {out_path}")
    else:
        grid = make_image_grid(images, rows=1, cols=len(images))
        grid.save(out_path)
        print(f"[OK] saved grid: {out_path}")

if __name__ == "__main__":
    main()