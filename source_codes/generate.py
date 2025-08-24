import json
import time
import requests
import subprocess
from pathlib import Path
from urllib.parse import urlparse, parse_qs


PROMPT_IMAGE = Path("prompt_img.json")
PROMPT_VIDEO = Path("prompt_vid.json")

FPS = 24
LEN_SEC = 5

def download(INPUT_FILE: str, PROMPT_ID, OUTPUT_DIR: str, FILE_TYPE: str, retry_interval: int = 3, max_wait: int = 1800, debug= False):
    elapsed = 0
    file_name = Path(INPUT_FILE).stem
    sp = Path(f"{OUTPUT_DIR}/{file_name}.{FILE_TYPE}")
    
    status_url = f"http://127.0.0.1:8188/history/{PROMPT_ID}"
    file_url = f"http://127.0.0.1:8188/view?filename={file_name}_00001_.{FILE_TYPE}&type=output"
    
    print(f"[INFO] 최대 {max_wait}초 대기.")
    print(status_url)

    if debug: return False
    else:
        # 부모 디렉토리 생성
        sp.parent.mkdir(parents=True, exist_ok=True)

        while elapsed < max_wait:
            response = requests.get(status_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            hist = data.get(PROMPT_ID, {})
            status = (hist.get("status") or {}).get("completed", False)

            if status:
                with open(sp, "wb") as f:
                    media = requests.get(file_url, stream=True, timeout=10)
                    media.raise_for_status()
                    for chunk in media.iter_content(1024):
                        f.write(chunk)

                print(f"[GOOD] 다운로드 완료 ({str(sp)})")
                return True

            elif elapsed + retry_interval >= max_wait:
                print("[ERROR] 다운로드 실패 (최대 대기 시간 초과)")
                return False
            
            else:
                print('.')
                time.sleep(retry_interval)
                elapsed += retry_interval
        return False

# 주어진 json 파일로부터 정보 가져오기
def read_json(INPUT_FILE):      # 파일 경로 -> (긍정 프롬, 부정 프롬, 다운로드 위치)
    p = Path(INPUT_FILE)
    with open(p, "r", encoding="utf-8") as f: data = json.load(f)
        
    pos = data.get("positive", None)
    neg = data.get("negative", None)
    return (pos, neg)


# 불러온 정보로부터 prompt_().json 파일 작성 (모델 용) 
def write_img_json(INPUT_FILE):     # 파일 경로 -> 다운로드 위치
    pos, neg = read_json(INPUT_FILE)
    with open(PROMPT_IMAGE, "r", encoding="utf-8") as f: data = json.load(f)
    
    data["prompt"]["9"]["inputs"]["filename_prefix"] = Path(INPUT_FILE).stem
    if pos: data["prompt"]["4"]["inputs"]["text"] = pos
    if neg: data["prompt"]["5"]["inputs"]["text"] = neg
    
    with open(PROMPT_IMAGE, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=4)

def write_vid_json(INPUT_FILE):     # 파일 경로 -> 다운로드 위치
    pos, neg = read_json(INPUT_FILE)
    with open(PROMPT_VIDEO, "r", encoding="utf-8") as f: data = json.load(f)

    stem = Path(INPUT_FILE).stem
    data["prompt"]["97"]["inputs"]["image"] = stem+'.png'
    data["prompt"]["108"]["inputs"]["filename_prefix"] = stem
    data["prompt"]["94"]["inputs"]["fps"] = FPS
    data["prompt"]["98"]["inputs"]["length"] = FPS*LEN_SEC
    
    if pos: data["prompt"]["93"]["inputs"]["text"] = pos
    if neg: data["prompt"]["89"]["inputs"]["text"] = neg

    with open(PROMPT_VIDEO, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=4)


def send_to_VM(INPUT_FILE):
    p = Path(INPUT_FILE).stem
    # 로컬(out_img)에서 생성된 이미지를 ComfyUI input으로 업로드
    # 터널링이 열려 있으므로 127.0.0.1:8188 로 POST
    root = Path(__file__).resolve().parents[1]  # AVA-Connect 루트
    local_img = (root / "out_img" / f"{p}.png").resolve()

    if not local_img.is_file(): print(f"로컬 이미지가 없습니다: {local_img}"); return False
    cmd = (
        f'curl -sS --fail -X POST '
        f'-F "image=@{local_img};filename={p}.png" '
        f'-F "type=input" -F "subfolder=" -F "overwrite=true" '
        f'http://127.0.0.1:8188/upload/image'
    )
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0: print(f"업로드 실패: {local_img}"); return False
    return True

# 비디오 생성 요청 및 다운로드
def fetch_video(INPUT_FILE, OUTPUT_DIR, debug= False):
    write_vid_json(INPUT_FILE)     # 파일 읽기, 전달 파일 작성, 다운로드 위치 설정
    
    OUTPUT_DIR = "../out_vid"
    
    did_send = send_to_VM(INPUT_FILE)
    
    if not debug:
        if did_send: 
            cmd = f'curl -s -X POST -H "Content-Type: application/json" -d @{PROMPT_VIDEO} http://127.0.0.1:8188/prompt | jq'
            curl_out = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True).stdout
            data = json.loads(curl_out)
            PROMPT_ID = data.get("prompt_id")
            print("[GOOD] 비디오 생성 요청 완료.")
            did_download = download(INPUT_FILE, PROMPT_ID, OUTPUT_DIR, 'mp4', debug= debug)
            return did_download
        else: 
            print("[SKIP] 전송 이미지가 없어 생성 요청을 건너뜁니다.")
            return False
    else: 
        print("[SKIP] 토큰 절약을 위해 디버깅 모드에서는 생성 요청을 생략합니다.")
        return False

# 비디오 생성 요청 및 다운로드
def fetch_image(INPUT_FILE, OUTPUT_DIR, debug= False):
    write_img_json(INPUT_FILE)     # 파일 읽기, 전달 파일 작성, 다운로드 위치 설정
    
    if not debug:
        print("[POST] 프롬프트 json을 전송합니다.")
        cmd = f'curl -s -X POST -H "Content-Type: application/json" -d @{PROMPT_IMAGE} http://127.0.0.1:8188/prompt'
        curl_out = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True).stdout
        data = json.loads(curl_out)
        PROMPT_ID = data.get("prompt_id")
        print("[GOOD] 이미지 생성 요청 완료.")
        did_download = download(INPUT_FILE, PROMPT_ID, OUTPUT_DIR, 'png', debug= debug)
        return did_download
    else: 
        print("[SKIP] 토큰 절약을 위해 디버깅 모드에서는 생성 요청을 생략합니다.")
        return False

# Queue 디렉토리 안의 모든 json에 대해 반복
def request_queue(content_type, debug= False):
    q_dir = "../q_"+content_type
    out_dir = "../out_"+content_type
    
    files = sorted(Path(q_dir).glob("*.json"))
    total, success, times = len(files), 0, []
    downloaded_files = []
    
    if not files:
        print(f"[INFO] 큐가 비어있습니다: {Path(q_dir).resolve()}")
        return times

    for file in files: 
        if debug: print(f"[DEBUG] {file}")
        did_download = False
        
        start_time = time.perf_counter()
        
        if content_type == 'img':
            did_download = fetch_image(INPUT_FILE= file, OUTPUT_DIR= out_dir, debug= debug)
        elif content_type == 'vid':
            did_download = fetch_video(INPUT_FILE= file, OUTPUT_DIR= out_dir, debug= debug)
            
        if did_download:
            downloaded_files.append(file)
            success += 1
            
        elapsed = time.perf_counter() - start_time
        times.append(elapsed)
        
    for file in downloaded_files:
        subprocess.run(f"mv {file} ../_trash/{file}", shell=True, check=True)
    
    # 큐 길이, 다운로드 수, 전체 시간
    return [total, success, times]