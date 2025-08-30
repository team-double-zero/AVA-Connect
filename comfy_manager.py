import time
import threading
import random
import requests
import datetime
from collections import deque
import copy

import vast_helper
from tunnel_manager import TunnelManager

default_img_prompt = {
    "prompt": {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_fp8_e4m3fn.safetensors", "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "L": {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["2", 0], "lora_name": "Qwen-Image-Lightning-8steps-V1.0.safetensors", "strength_model": 0.8, "strength_clip": 0.8}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "some positive prompts", "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, drawing, same face, identical face, generic face, repetitive face", "clip": ["2", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 1080, "height": 1920, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"seed": 383255100, "steps": 28, "cfg": 5.5, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": ["L", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "image/test1", "overwrite": True, "images": ["8", 0]}}
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
            "text": "some positive prompts", "clip": ["84", 0]}},
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
            "filename_prefix": "video/test1", "format": "mp4", "codec": "h264", "overwrite": True, "video": ["94", 0]}}
    }
}

class ComfyQueueItem:
    """QueueManager 큐에 들어갈 객체 클래스, 이미지 및 비디오 생성 규칙에 맞춰 data 딕셔너리를 반환합니다."""
    
    prompt_id = None
    download_url = None
    
    DEFAULTS = {
        "file_name": f"file_{datetime.datetime.now():%m%d_%H%M%S}",    # 파일 이름, 기본값 = file_날짜_시간
        "width": 1080,              # 이미지 및 영상 사이즈
        "height": 1920,
        "length": 5,                # 영상 길이 (sec)
        "seed": 0,                  # 랜덤 시드
        "negative": "default",      # 부정 프롬프트
        # "positive" prompt (필수 인자)
        # "content_type" (필수 인자) - image / video
    }

    def __init__(self, **kwargs):
        """body {"positive", "content_type", ...}"""
        
        """body 내용의 필수 인자를 검사합니다."""
        required = ["content_type", "positive"]
        for key in required:
            if key not in kwargs or not kwargs:
                raise SyntaxError(f"필수 입력 누락: {key}")

        """입력된 내용을 기본 내용에 덮어 씌웁니다."""
        full_body = {**ComfyQueueItem.DEFAULTS, **kwargs}

        content_type = full_body["content_type"]
        file_name = full_body["file_name"]
        ext = 'png' if content_type == 'image' else 'mp4'
        self.download_url = f"http://127.0.0.1:8080/view?filename={file_name}_00001_.{ext}&type=output&subfolder={content_type}"

        """파일 형식에 맞는 프롬프트 데이터를 self.data에 작성합니다."""
        if content_type == 'image':
            self.data = copy.deepcopy(default_img_prompt)
            self.data["prompt"]["9"]["inputs"]["filename_prefix"] = 'png/'+full_body["file_name"]+'.png'
            self.data["prompt"]["7"]["inputs"]["seed"] = random.randint(0, 2**30)
            
            self.data["prompt"]["4"]["inputs"]["text"] = full_body["positive"]
            if full_body["negative"] != "default": self.data["prompt"]["5"]["inputs"]["text"] = full_body["negative"]

        elif content_type == 'video':
            self.data = copy.deepcopy(default_vid_prompt)
            self.data["prompt"]["108"]["inputs"]["filename_prefix"] = 'png/'+full_body["file_name"]+'.mp4'
            self.data["prompt"]["97"]["inputs"]["image"] = full_body["file_name"]+'.png'
            
            self.data["prompt"]["94"]["inputs"]["fps"] = 24
            self.data["prompt"]["98"]["inputs"]["seed"] = full_body["length"] * 24
            
            self.data["prompt"]["93"]["inputs"]["text"] = full_body["positive"]
            if full_body["negative"] != "default": self.data["prompt"]["89"]["inputs"]["text"] = full_body["negative"]

    def __str__(self) -> str:
        """data 문자열을 반환합니다."""
        return str(self.data)

    def __dict__(self) -> dict:
        """data 딕셔너리를 반환합니다."""
        return self.data


class ComfyManager():
    LOOP_INTERVAL = 5 # 루프 간격(초)
    IDLE_TIMEOUT = 30 # idle 상태 타임아웃(초)
    WAIT_THRESHOLD = 1 # wait queue 임계점

    _instance = None
    _tunnelManager:TunnelManager = None
    _vast_instance:vast_helper.VastHelper = None
    
    wait_queue = deque()    # 기본 ComfyRequest 딕셔너리
    working_queue = deque() # + 조회 용 prompt_id
    output_queue = deque()
    error_queue = deque()
    
    error_message:str = None
    
    _idle_time:int = 0 # idle로 유지된 시간
    
    _init_thread = None
    """ComfyUI 초기화 쓰레드"""
    
    _loop_thread = None
    """내부 루프 쓰레드"""
    
    _status:str = "init"
    """
    ComfyManager의 상태: init, connecting, idle, working, error
    - init: 초기화
    - booting: vast.ai instance 부팅 대기
    - connecting: 연결 중
    - idle: 대기 -> 5분 초과 시 인스턴스 자동 종료
    - working: 작업 중
    - error: 에러. 메시지는 [error_message]로 저장됩니다.
    """

    def __new__(cls, *args, **kwargs):
        """싱글톤: 모든 객체에 대해 같은 인스턴스를 반환합니다."""
        if cls._instance is None:
            cls._instance = super(ComfyManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, local_port: int):
        self.local_port:int = local_port
        # 내부 쓰레드 루프 시작
        self._loop_thread = threading.Thread(target=self._loop)
        self._loop_thread.start()

    @property
    def status(self) -> str:
        return self._status

    def _init_comfyui(self):
        try:
            self._status = "booting"
            self._vast_helper = vast_helper.VastHelper()
            host, port, best_instance = self._vast_helper.run_best_instance()
            
            self._status = "connecting"
            self._vast_instance = best_instance
            
            time.sleep(3)
            
            self._tunnelManager = TunnelManager(host, port)
            self._tunnelManager.run_comfyui()
            
            connected = self._tunnelManager.check_comfyui_connection()

            if connected:
                self._status = "idle"
            else:
                self._status = "error"
                self.error_message = "can't connect to ComfyUI"        
            
        except Exception as e:
            self._status = "error"
            self.error_message = f"while init: {e}"

    def _loop(self):
        while self.status != "error" and self.status != None:
            # run code
            print(f"[Thread] loop running status={self.status} idle_time{self._idle_time}")

            try:
                # ComfyUI 비동기 초기화
                if self.status == "init" and len(self.wait_queue) >= self.WAIT_THRESHOLD:
                    self._init_thread = threading.Thread(target=self._init_comfyui)
                    self._init_thread.start()

                # ComfyUI 요청하기
                # if (self.status == "idle" or self.status == "working") and self.wait_queue:
                if (self.status == "idle" or self.status == "working"):
                    while self.wait_queue:
                        wait_item = self.wait_queue.popleft()
                        self._request_image_gen(wait_item) # -> working_queue
                        self._status = "working" if len(self.working_queue) > 0 else "idle"
                    
                # ComfyUI 조회하기 -> output_queue
                if self.status == "working":
                    # 조회 -> 새로운 히스토리가 나타날 경우 output_queue에 삽입
                    self._handle_comfy_working_queue()
                    if len(self.working_queue) == 0:
                        self._status = "idle"
                
                time.sleep(self.LOOP_INTERVAL)
                
                # idle time
                if self.status == "idle":
                    self.idle_time += self.LOOP_INTERVAL
                else:
                    self.idle_time = 0

                # idle -> init
                if self.idle_time > self.IDLE_TIMEOUT:
                    self._vast_helper.stop_instance(self._vast_instance)
                    self._vast_instance = None
                    self._status = "init"
            
            except Exception as e:
                self._status = "error"
                self.error_message = f"error while thread loop {e}"

        # 쓰레드 종료
        print(f"[Thread] End {self.status} {self.error_message}")

    def _request_image_gen(self, wait_item: ComfyQueueItem):
        """이미지 생성을 요청합니다."""
        # print("[POST] 이미지 생성을 요청합니다.")
        
        try:
            url = f"http://127.0.0.1:{self.local_port}/prompt"
            response = requests.post(url, json= wait_item.data)
            data = response.json()
            
            prompt_id = data.get("prompt_id")
            wait_item.prompt_id = prompt_id
            print(prompt_id)
            
            self.working_queue.append(wait_item)
            
        except Exception as e:
            print(f"[ERROR] error whiel _request_image_gen {e}")
            wait_item.data["error"] = str(e)
            self.error_queue.append(wait_item)

    # def fetch_comfy_queue(self) -> list[ComfyQueueItem]:
    #     queue_url = f"http://127.0.0.1:{self.local_port}/queue"
    #     resp = requests.get(queue_url, timeout=10)
    #     data = resp.json()
    #     print(f"comfy_queue_data={resp.json()}")
    #     return []
    
    def _handle_comfy_working_queue(self) -> None:
        history_url = f"http://127.0.0.1:{self.local_port}/history"
        resp = requests.get(history_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # print(f"comfy_history_data={resp.json()}"[:500])

        temp_queue = deque()
        while self.working_queue:
            item = self.working_queue.popleft()
            
            if hasattr(item.prompt_id, "prompt_id") or not item.prompt_id:
                temp_queue.append(item)
                continue
            
            # wait_item이 history에 존재하는지 조회
            prompt_data = data.get(item.prompt_id)
            if prompt_data is None:
                temp_queue.append(item)
                continue

            # 히스토리에 존재함 -> status 조회
            prompt_status = prompt_data.get("status_str")
            if prompt_status == "success":
                self.output_queue.append(item)
                continue
            elif prompt_status == "error":
                error_message = prompt_data.get("error")
                print(f"error from ComfyUI(prompt_id={item.prompt_id}): {error_message}")
                self.error_queue.append(item)
                continue
            else:
                print("unknown status", prompt_status)
                temp_queue.append(item)

        while temp_queue:
            self.working_queue.append(temp_queue.popleft())


    def push_wait(self, q_item: ComfyQueueItem):
        """대기큐 아이템을 하나 삽입합니다."""
        self.wait_queue.append(q_item)

    def output_pop(self) -> ComfyQueueItem:
        """출력큐 아이템을 하나 반환합니다."""
        if self.output_queue:
            item = self.output_queue.popleft()
            return item
        return None
    
    def has_output(self) -> bool:
        return len(self.output_queue) > 0

def test_case_1():
    print("queue controller.py debug")
    
    qm = ComfyManager(local_port= 8080)
    
    # 첫 번째 아이템 삽입
    image_req_1 = ComfyQueueItem(
        positive = "Korean sexy girl, wearing bikini, dancing on the beach, charming body",
        content_type = "image"
    )
    qm.push_wait(image_req_1)
    
    time.sleep(1)
    
    while qm.status == "init" or qm.status == "booting" or qm.status == "connecting":
        time.sleep(1)

    # 두 번째 아이템 삽입
    image_req_2 = ComfyQueueItem(
        positive = "Korean sexy girl, wearing school uniform, dancing on the classroom, charming body",
        content_type = "image"
    )
    qm.push_wait(image_req_2)

    while qm.status != "idle" and qm.status != "error":
        time.sleep(1)

    expected_count = 2
    outputs = []

    while expected_count > 0:
        if qm.has_output():
            expected_count -= 1
            outputs.append(qm.output_pop())
        else:
            time.sleep(1)

    print('\n'.join(outputs))



if __name__ == "__main__":
    test_case_1()