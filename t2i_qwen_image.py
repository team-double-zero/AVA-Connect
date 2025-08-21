import json
from diffusers import DiffusionPipeline
import torch

# 1. JSON 불러오기
with open("qwen_test.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

# 2. 모델 로드 (이미 ComfyUI/models 하위 폴더 옮겼다고 했으니 경로 맞게)
pipe = DiffusionPipeline.from_pretrained(
    "./models/diffusion_models/qwen",
    torch_dtype=torch.float16
).to("cuda")

# 3. JSON에서 옵션 꺼내기
prompt = cfg.get("prompt", "A photo")
negative_prompt = cfg.get("negative_prompt", "")
width = cfg.get("width", 512)
height = cfg.get("height", 512)
steps = cfg.get("num_inference_steps", 25)
guidance = cfg.get("guidance_scale", 7.5)
seed = cfg.get("seed", None)

generator = torch.manual_seed(seed) if seed is not None else None

# 4. 이미지 생성
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=steps,
    guidance_scale=guidance,
    generator=generator
).images[0]

# 5. 저장
image.save("qwen_out.png")
print("✅ Generated image saved as qwen_out.png")