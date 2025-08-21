# t2i_qwen.py
import argparse
from pathlib import Path
import sys
import torch
from PIL import Image

def _fail(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", default="/root/AVA-Connect/models",
                    help="모델 루트 폴더 (loras/ vae/ diffusion_models/ text_encoders/ 포함)")
    ap.add_argument("--prompt", required=True, help="생성 프롬프트")
    ap.add_argument("--n_prompt", default="worst quality, low quality, blurry, artifacts")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num", type=int, default=1)
    ap.add_argument("--lora", default=None, help="선택: LoRA safetensors 경로")
    ap.add_argument("--out", default="/root/AVA-Connect/outputs/qwen_out.png")
    args = ap.parse_args()

    mr = Path(args.model_root)
    dm = mr / "diffusion_models" / "qwen_image_fp8_e4m3fn.safetensors"
    vae = mr / "vae" / "qwen_image_vae.safetensors"
    te  = mr / "text_encoders" / "qwen_2.5_vl_7b_fp8_scaled.safetensors"

    print("[INFO] expecting files:")
    print(f"  UNet (ckpt?) : {dm}")
    print(f"  VAE          : {vae}")
    print(f"  TextEncoder  : {te}")
    if args.lora:
        print(f"  LoRA         : {args.lora}")

    # 존재 체크
    miss = [p for p in [dm, vae, te] if not p.exists()]
    if miss:
        _fail("필수 파일이 없습니다: " + ", ".join(str(m) for m in miss))

    # 1) from_single_file 로 먼저 시도 (이 파일이 '완전한' 단일 ckpt일 때만 성공)
    try:
        from diffusers import AutoPipelineForText2Image
        from diffusers.utils import make_image_grid
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("[INFO] trying diffusers.from_single_file(...) on qwen_image_fp8_e4m3fn.safetensors ...")
        pipe = AutoPipelineForText2Image.from_single_file(
            dm.as_posix(),
            torch_dtype=dtype
        )
        pipe = pipe.to(device)

        # (선택) LoRA 적용 시도
        if args.lora:
            try:
                pipe.load_lora_weights(args.lora)
                try:
                    pipe.fuse_lora()
                except Exception:
                    pass
                print("[INFO] LoRA loaded.")
            except Exception as e:
                print(f"[WARN] LoRA 로드 실패 (무시): {e}")

        g = torch.Generator(device=device)
        if args.seed is not None:
            g = g.manual_seed(args.seed)

        print("[INFO] generating...")
        images = pipe(
            prompt=args.prompt,
            negative_prompt=args.n_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            width=args.width,
            height=args.height,
            num_images_per_prompt=args.num,
            generator=g
        ).images

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if len(images) == 1:
            images[0].save(out_path)
        else:
            grid = make_image_grid(images, rows=1, cols=len(images))
            grid.save(out_path)
        print(f"[OK] saved: {out_path}")
        return

    except Exception as e:
        print(f"[WARN] from_single_file 로드는 실패했습니다: {e}")

    # 2) 여기로 오면, qwen_image_fp8_e4m3fn.safetensors 가 '단일 SD류 체크포인트'가 아님.
    #    즉, UNet/TE/VAE가 분리되어 있고 diffusers 포맷 디렉터리가 필요하거나,
    #    해당 모델의 '공식 실행 스크립트'를 통해 로드해야 합니다.
    print("\n[GUIDE] 현재 파일 구성은 분리된 UNet/VAE/TextEncoder 형태로 보입니다.")
    print("        diffusers AutoPipeline은 단일 체크포인트이거나, 완전한 diffusers 폴더(model_index.json 포함)가 필요합니다.")
    print("        아래 두 가지 중 하나를 선택하세요:\n")
    print("   A) (권장) 모델을 'diffusers 포맷 디렉터리'로 준비해 이 스크립트의 --model_root를 그 디렉터리로 지정")
    print("      - 즉, model_index.json / unet/ vae/ text_encoder/ ... 하위 폴더가 있는 구조")
    print("   B) Qwen 이미지 모델의 '공식 실행 스크립트/레포'를 사용해 T2I를 수행")
    print("\n[HINT] 현재 경로들:")
    print(f"      - diffusion ckpt: {dm}")
    print(f"      - vae          : {vae}")
    print(f"      - text encoder : {te}")
    print("\n[NOTE] 필요하시면, 위 세 파일을 기반으로 diffusers 포맷 디렉터리 스캐폴딩을 자동 생성하는 보조 스크립트를 만들어 드릴게요.")
    sys.exit(2)

if __name__ == "__main__":
    main()