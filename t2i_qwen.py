import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", default="/root/AVA-Connect/models")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--n_prompt", default="worst quality, low quality, blurry, artifacts")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--out", default="qwen_out.png")
    args = parser.parse_args()

    mr = Path(args.model_root)

    # 경로 지정
    unet_path = mr / "diffusion_models" / "qwen_image_fp8_e4m3fn.safetensors"
    vae_path = mr / "vae" / "qwen_image_vae.safetensors"
    te_path  = mr / "text_encoders" / "qwen_2.5_vl_7b_fp8_scaled.safetensors"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("[INFO] Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path=None,
        state_dict=torch.load(unet_path, map_location="cpu"),
        torch_dtype=dtype,
    )

    print("[INFO] Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path=None,
        state_dict=torch.load(vae_path, map_location="cpu"),
        torch_dtype=dtype,
    )

    print("[INFO] Loading Text Encoder...")
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path=None,
        state_dict=torch.load(te_path, map_location="cpu"),
        torch_dtype=dtype,
    )

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5")

    print("[INFO] Building pipeline...")
    pipe = StableDiffusionPipeline(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)

    g = torch.Generator(device=device)
    if args.seed is not None:
        g.manual_seed(args.seed)

    print("[INFO] Generating...")
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
    images[0].save(out_path)
    print(f"[OK] Saved at {out_path}")

if __name__ == "__main__":
    main()