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
    output_url = None
    
    DEFAULTS = {
        "width": 1080,              # 이미지 및 영상 사이즈
        "height": 1920,
        "length": 5,                # 영상 길이 (sec)
        "seed": 0,                  # 랜덤 시드
        "negative": "text, logo, watermark, too many watermarks, blank page, text-only page, reference, username, signature, artist:xinzoruo, artist:milkpanda, artist collaboration, variant set, large variant set, 4koma, 2koma, toon (style), oekaki, chibi, turnaround, film grain, monochrome, dithering, halftone, screentones, dated, old, 1990s (style), mutation, deformed, distorted, disfigured, artistic error, distorted anatomy, anatomical structure error, asymmetrical face, unnatural hair, bad eyes, cloudy eyes, blank eyes, pointy ears, bad proportions, bad limb, bad hands, extra hands, bad hand structure, extra digits, fewer digits, bad legs, extra legs, amputee, distorted composition, bad perspective, multiple views, negative space, animation error, chromatic aberration, disorganized colors, scan artifacts, jpeg artifacts, vertical lines, vertical banding, worst quality, bad quality, lowres, blurry, upscaled, fewer details, unfinished, incomplete, amateur, cheesy, unsatisfactory, inadequate, deficient, subpar, poor, displeasing, very displeasing, bad illustration, bad portrait, animal ear, cat ears, dark, dark hole",      # 부정 프롬프트
        # "positive" prompt (필수 인자)
        # "content_type" (필수 인자) - image / video
        # "file_name" - 인스턴스 생성 시 동적으로 생성
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
        
        # file_name이 제공되지 않았다면 현재 시간으로 생성 (고유한 파일명 보장)
        if "file_name" not in full_body:
            import time
            timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
            microsec = int(time.time() * 1000000) % 1000  # 마이크로초 추가로 고유성 보장
            full_body["file_name"] = f"file_{timestamp}_{microsec}"

        content_type = full_body["content_type"]
        file_name = full_body["file_name"]
        ext = 'png' if content_type == 'image' else 'mp4'
        # 로컬 포트를 명시적으로 받아서 사용
        local_port = full_body.get("local_port", 8080)
        self.comfy_url = f"http://127.0.0.1:{local_port}"
        self.output_url = f"{self.comfy_url}/view?filename={file_name}_00001_.{ext}&type=output&subfolder={content_type}"
        
        # 초기 output_url 생성 (나중에 실제 파일명으로 업데이트됨)
        # print(f"[DEBUG] 초기 output_url: {self.output_url}")

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
        return str({**self.data, "output_url": self.output_url, "prompt_id": self.prompt_id})

    def __dict__(self) -> dict:
        """data 딕셔너리를 반환합니다."""
        return {**self.data, "output_url": self.output_url, "prompt_id": self.prompt_id}
    
    def download(self) -> bytes:
        """output_url을 이용해 생성된 파일을 다운로드하여 bytes로 반환합니다."""
        if not self.output_url:
            raise ValueError("output_url이 설정되지 않았습니다.")
        
        try:
            response = requests.get(self.output_url, timeout=30, stream=True)
            response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
            
            # 바이트 데이터로 수집
            content = b''
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
            
            return content
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"다운로드 실패 ({self.output_url}): {e}")

    def update_output_url_from_history(self, outputs: dict):
        """ComfyUI history의 outputs에서 실제 파일명을 추출하여 output_url 업데이트"""
        try:
            # outputs에서 실제 파일 정보 찾기
            actual_filename = None
            subfolder = ""
            file_type = "output"
            
            for node_id, output in outputs.items():
                if "images" in output and output["images"]:
                    # 이미지 파일
                    image_info = output["images"][0]  # 첫 번째 이미지
                    actual_filename = image_info["filename"]
                    subfolder = image_info.get("subfolder", "")
                    file_type = image_info.get("type", "output")
                    print(f"[DEBUG] 이미지 파일 발견: {actual_filename}")
                    break
                elif "gifs" in output and output["gifs"]:
                    # 비디오/GIF 파일
                    gif_info = output["gifs"][0]  # 첫 번째 파일
                    actual_filename = gif_info["filename"]
                    subfolder = gif_info.get("subfolder", "")
                    file_type = gif_info.get("type", "output")
                    print(f"[DEBUG] 비디오/GIF 파일 발견: {actual_filename}")
                    break
            
            if actual_filename:
                # 실제 파일명으로 output_url 업데이트
                if subfolder:
                    self.output_url = f"{self.comfy_url}/view?filename={actual_filename}&type={file_type}&subfolder={subfolder}"
                else:
                    self.output_url = f"{self.comfy_url}/view?filename={actual_filename}&type={file_type}"
                print(f"[DEBUG] output_url 업데이트: {self.output_url}")
            else:
                print(f"[WARNING] outputs에서 파일을 찾을 수 없음: {outputs}")
                
        except Exception as e:
            print(f"[ERROR] output_url 업데이트 실패: {e}")
            # 기존 URL 유지


class ComfyManager():
    LOOP_INTERVAL = 5 # 루프 간격(초)
    IDLE_TIMEOUT = 300 # idle 상태 타임아웃(초)
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
            
            self._tunnelManager = TunnelManager(host, port, local_port=self.local_port)
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
            print(f"[Thread] loop running status={self.status} idle_time={self._idle_time} wait_queue={len(self.wait_queue)} working_queue={len(self.working_queue)} output_queue={len(self.output_queue)} error_queue={len(self.error_queue)}")

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
                    self._idle_time += self.LOOP_INTERVAL
                else:
                    self._idle_time = 0

                # idle -> init
                if self._idle_time > self.IDLE_TIMEOUT:
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
        """견고한 ComfyUI 작업 큐 처리"""
        try:
            history_url = f"http://127.0.0.1:{self.local_port}/history"
            resp = requests.get(history_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[ERROR] 히스토리 조회 실패: {e}")
            return  # 조회 실패 시 다음 루프에서 재시도

        temp_queue = deque()
        processed_count = 0
        
        while self.working_queue and processed_count < 50:  # 무한 루프 방지
            item = self.working_queue.popleft()
            processed_count += 1
            
            # prompt_id 유효성 검사
            if not hasattr(item, 'prompt_id') or not item.prompt_id:
                temp_queue.append(item)
                continue
            
            # 히스토리에서 해당 프롬프트 데이터 조회
            prompt_data = data.get(item.prompt_id)
            if prompt_data is None:
                # 아직 히스토리에 없음 - 처리 중일 가능성
                temp_queue.append(item)
                continue

            # 상태 정보 추출 (여러 필드 체크)
            status_info = self._extract_prompt_status(prompt_data)
            
            if status_info["status"] == "success":
                print(f"[SUCCESS] 작업 완료: {item.prompt_id}")
                # 실제 파일명으로 output_url 업데이트
                if "outputs" in status_info:
                    item.update_output_url_from_history(status_info["outputs"])
                self.output_queue.append(item)
                continue
                
            elif status_info["status"] == "error":
                error_msg = status_info.get("error_message", "알 수 없는 오류")
                print(f"[ERROR] ComfyUI 작업 실패 (prompt_id={item.prompt_id}): {error_msg}")
                item.data["error"] = error_msg
                self.error_queue.append(item)
                continue
                
            elif status_info["status"] == "processing":
                # 여전히 처리 중
                temp_queue.append(item)
                continue
                
            else:
                # 알 수 없는 상태 처리
                current_retry_count = getattr(item, '_retry_count', 0)
                if current_retry_count < 3:  # 최대 3회 재시도
                    item._retry_count = current_retry_count + 1
                    debug_info = status_info.get('debug_info', {})
                    print(f"[RETRY] 알 수 없는 상태 재시도 ({item._retry_count}/3)")
                    print(f"  - prompt_id: {item.prompt_id}")
                    print(f"  - raw_status: {status_info['raw_status']}")
                    print(f"  - debug_info: {debug_info}")
                    temp_queue.append(item)
                else:
                    print(f"[TIMEOUT] 상태 확인 실패로 에러 처리: {item.prompt_id}")
                    print(f"  - 최종 상태: {status_info['raw_status']}")
                    print(f"  - 디버그: {status_info.get('debug_info', {})}")
                    item.data["error"] = f"상태 확인 실패: {status_info['raw_status']}"
                    self.error_queue.append(item)

        # 처리되지 않은 아이템들을 다시 큐에 추가
        while temp_queue:
            self.working_queue.append(temp_queue.popleft())

    def _extract_prompt_status(self, prompt_data: dict) -> dict:
        """프롬프트 데이터에서 상태 정보 추출 (실제 ComfyUI API 구조에 맞춤)"""
        
        # 상태 정보는 status 객체 안에 중첩됨
        status_obj = prompt_data.get("status", {})
        
        # 여러 상태 필드 체크 (실제 API 구조에 맞춰 수정)
        status_str = status_obj.get("status_str")  # 핵심 상태 필드
        completed = status_obj.get("completed", False)  # 완료 여부
        
        # outputs 필드로 실제 결과 확인
        outputs = prompt_data.get("outputs", {})
        has_outputs = bool(outputs)
        
        # 오류 정보 추출 (여러 위치에서 확인)
        error_info = (
            status_obj.get("error") or
            prompt_data.get("error") or 
            prompt_data.get("exception") or
            status_obj.get("exception")
        )
        
        retry_count = getattr(prompt_data, '_retry_count', 0)
        
        # 상태 결정 로직 (실제 ComfyUI 동작에 맞춤)
        if status_str == "success" and completed and has_outputs:
            return {
                "status": "success",
                "retry_count": retry_count,
                "raw_status": status_str,
                "outputs": outputs
            }
        elif status_str == "error" or error_info:
            return {
                "status": "error", 
                "error_message": str(error_info) if error_info else "Unknown error",
                "retry_count": retry_count,
                "raw_status": status_str
            }
        elif status_str in ["running", "executing", "pending", "queued"] or not completed:
            return {
                "status": "processing",
                "retry_count": retry_count, 
                "raw_status": status_str
            }
        elif status_str is None and not completed and not has_outputs:
            # 아직 처리 시작되지 않음
            return {
                "status": "processing",
                "retry_count": retry_count,
                "raw_status": "pending"
            }
        else:
            # 알 수 없는 상태
            return {
                "status": "unknown",
                "retry_count": retry_count,
                "raw_status": status_str,
                "debug_info": {
                    "status_str": status_str,
                    "completed": completed,
                    "has_outputs": has_outputs,
                    "status_obj": status_obj
                }
            }


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
    from dotenv import load_dotenv
    load_dotenv()
    print("test_case_1()")
    
    qm = ComfyManager(local_port= 8090)
    
    # 첫 번째 아이템 삽입
    image_req_1 = ComfyQueueItem(
        positive = "1girl, split screen, two views, left_and_right, clothed left nude right, character profile, {{same pose}}, same position, pubic hair, small breasts, pussy, nipples, nude, navel, blush, ass visible through thighs, standing, indoors, looking at viewer, nsfw, best quality, amazing quality, very aesthetic, highres, incredibly absurdres",
        content_type = "image",
        local_port = 8090
    )
    qm.push_wait(image_req_1)
    
    time.sleep(1)
    
    while qm.status == "init" or qm.status == "booting" or qm.status == "connecting":
        time.sleep(1)

    # 두 번째 아이템 삽입
    image_req_2 = ComfyQueueItem(
        positive = "Korean sexy girl, wearing school uniform, dancing on the classroom, charming body",
        content_type = "image",
        local_port = 8090
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

    # 결과물 출력 및 다운로드
    print("\n=== 생성된 이미지 정보 ===")
    for i, output in enumerate(outputs, 1):
        if output:
            print(f"이미지 {i}:")
            print(f"  Prompt ID: {output.prompt_id}")
            print(f"  Output URL: {output.output_url}")
            print()
    
    # outputs 폴더 생성 (없으면 생성)
    import os
    os.makedirs("outputs", exist_ok=True)
    
    # 각 이미지를 다운로드하여 파일로 저장
    print("=== 이미지 다운로드 시작 ===")
    for i, output in enumerate(outputs, 1):
        if output:
            try:
                print(f"[DOWNLOAD] 이미지 {i} 다운로드 중...")
                image_bytes = output.download()
                
                # 파일명 생성 (outputs 폴더 아래)
                filename = f"outputs/downloaded_image_{output.prompt_id}_{i}.png"
                
                # 파일로 저장
                with open(filename, 'wb') as f:
                    f.write(image_bytes)
                
                print(f"[SUCCESS] 저장 완료: {filename} ({len(image_bytes):,} bytes)")
                
            except Exception as e:
                print(f"[ERROR] 이미지 {i} 다운로드 실패: {e}")
    
    print("\n=== 다운로드 완료 ===")
    print("다운로드된 파일들을 확인해보세요!")


if __name__ == "__main__":
    test_case_1()