import os, random, datetime

from dotenv import load_dotenv
from vastai_sdk import VastAI
from collections import deque

default_img_prompt = {
    "prompt": {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_fp8_e4m3fn.safetensors", "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "L": {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["2", 0], "lora_name": "Qwen-Image-Lightning-8steps-V1.0.safetensors", "strength_model": 0.8, "strength_clip": 0.8}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "cute asian idol girl, wearing tight gray maxi dress, big tits, full body, random face", "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, drawing, same face, identical face, generic face, repetitive face", "clip": ["2", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 1080, "height": 1920, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"seed": 383255100, "steps": 28, "cfg": 5.5, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": ["L", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "png/test1", "overwrite": True, "images": ["8", 0]}}
    }
}

default_vid_prompt = {
    "prompt": {
        "84": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "85": {"class_type": "KSamplerAdvanced", "inputs": {
            "add_noise": "disable", "noise_seed": 1, "steps": 4, "cfg": 1,
            "sampler_name": "euler", "scheduler": "simple", "start_at_step": 2, "end_at_step": 4,
            "return_with_leftover_noise": "disable", "model": ["103", 0],
            "positive": ["98", 0], "negative": ["98", 1], "latent_image": ["86", 0]}},
        "86": {"class_type": "KSamplerAdvanced", "inputs": {
            "add_noise": "disable", "noise_seed": 1, "steps": 4, "cfg": 1,
            "sampler_name": "euler", "scheduler": "simple", "start_at_step": 0, "end_at_step": 2,
            "return_with_leftover_noise": "enable", "model": ["104", 0],
            "positive": ["98", 0], "negative": ["98", 1], "latent_image": ["98", 2]}},
        "87": {"class_type": "VAEDecode", "inputs": {"samples": ["85", 0], "vae": ["90", 0]}},
        "89": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "low quality, blurry, trailing artifacts, motion blur, ghosting, transparent, overlapping, afterimage, watermark, text, logo",
            "clip": ["84", 0]}},
        "90": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "93": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "Sexy asian girl, dancing attractively, sexy, charming", "clip": ["84", 0]}},
        "94": {"class_type": "CreateVideo", "inputs": {"fps": 24, "images": ["87", 0]}},
        "95": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "96": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "97": {"class_type": "LoadImage", "inputs": {"image": "test1.png"}},
        "98": {"class_type": "WanImageToVideo", "inputs": {
            "width": 720, "height": 1080, "length": 120, "batch_size": 1,
            "positive": ["93", 0], "negative": ["89", 0], "vae": ["90", 0], "start_image": ["97", 0]}},
        "101": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
            "strength_model": 1.0, "model": ["95", 0]}},
        "102": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
            "strength_model": 1.0, "model": ["96", 0]}},
        "103": {"class_type": "ModelSamplingSD3", "inputs": {"shift": 5.0, "model": ["102", 0]}},
        "104": {"class_type": "ModelSamplingSD3", "inputs": {"shift": 5.0, "model": ["101", 0]}},
        "108": {"class_type": "SaveVideo", "inputs": {
            "filename_prefix": "mp4/test1", "format": "mp4", "codec": "h264", "overwrite": True, "video": ["94", 0]}}
    }
}

class GpuQueueItem:
    DEFAULTS = {
        "file_name": f"file_{datetime.datetime.now():%m%d_%H%M%S}",
        "width": 1080,
        "height": 1920,
        "length": 5,
        "seed": 0,
        "negative": "default",
    }

    def __init__(self, body: dict):
        required = ["content_type", "positive"]
        for key in required:
            if key not in body or not body[key]:
                raise SyntaxError(f"필수 입력 누락: {key}")

        full_body = {**GpuQueueItem.DEFAULTS, **body}
        self.full_body = full_body
        self.ext = full_body["content_type"]

        if self.ext == 'image':
            self.data = default_img_prompt.copy()
            self.data["prompt"]["9"]["inputs"]["filename_prefix"] = 'png/'+full_body["file_name"]+'.png'
            self.data["prompt"]["7"]["inputs"]["seed"] = 0 if full_body["seed"]==0 else random.randint(1, 2**30)
            
            self.data["prompt"]["4"]["inputs"]["text"] = full_body["positive"]
            if full_body["negative"] != "default": self.data["prompt"]["5"]["inputs"]["text"] = full_body["negative"]

        elif self.ext == 'video':
            self.data = default_vid_prompt.copy()
            self.data["prompt"]["108"]["inputs"]["filename_prefix"] = 'png/'+full_body["file_name"]+'.mp4'
            self.data["prompt"]["97"]["inputs"]["image"] = full_body["file_name"]+'.png'
            
            self.data["prompt"]["94"]["inputs"]["fps"] = 24
            self.data["prompt"]["98"]["inputs"]["seed"] = full_body["length"] * 24
            
            self.data["prompt"]["93"]["inputs"]["text"] = full_body["positive"]
            if full_body["negative"] != "default": self.data["prompt"]["89"]["inputs"]["text"] = full_body["negative"]


    def show_rough_data(self): 
        print(self.data)

    def __repr__(self):
        return f"<GpuQueueItem: body= {str(self.full_body)}>"


class GpuManager():
    _instance = None
    wait_queue = deque()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GpuManager, cls).__new__(cls)
        return cls._instance

    def q_status(self):
        l = len(self.wait_queue)
        print(f"[INFO] 큐에 {l}개의 항목이 대기중입니다.\n")
        for i, q in enumerate(list(self.wait_queue)):
            print(f"{i+1:02}. {q}")
            print()
        return l

    def q_add(self, item: GpuQueueItem):
        self.wait_queue.append(item)

    def q_pop(self):
        q = self.wait_queue.popleft()
        return q

    def q_empty(self):
        self.wait_queue.clear()


if __name__ == "__main__":
    gpuManager = GpuManager()
    
    gpuManager.q_add(GpuQueueItem(
        body={
            "file_name": "test_1",
            "content_type": "video",
            "positive": "some positive prompts"
        }))
    
    gpuManager.q_add(GpuQueueItem(
        body={
            "file_name": "test_2",
            "content_type": "image",
            "positive": "some positive prompts"
        }))
    
    gpuManager.q_add(GpuQueueItem(
        body={
            "content_type": "video",
            "positive": "some positive prompts"
        }))


    gm = GpuManager()
    gm.q_status()