import json, requests
import time
import subprocess
import sys, os, re
from pathlib import Path
from urllib.parse import urlparse, parse_qs

def download_image(url: str, save_path: str, retry_interval: int = 3, max_wait: int = 600):
    elapsed = 0
    attempt = 1
    print(f"[INFO] 대기 완료. 다운로드를 시도합니다.")
    print(url)

    # 만약 save_path가 디렉토리 경로로 들어오면, URL의 filename 파라미터를 파일명으로 사용
    sp = Path(save_path)
    if sp.exists() and sp.is_dir():
        try:
            q = parse_qs(urlparse(url).query)
            fname = q.get("filename", ["download.png"])[0]
        except Exception:
            fname = "download.png"
        sp = sp / fname
    # 부모 디렉토리 생성
    sp.parent.mkdir(parents=True, exist_ok=True)

    while elapsed < max_wait:
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()

            with open(sp, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)

            print(f"[GOOD] 다운로드 완료 ({str(sp)})")
            return True

        except Exception as e:
            if elapsed + retry_interval >= max_wait:
                print("[ERROR] 다운로드 실패 (최대 대기 시간 초과)")
                return False
            print('.')
            time.sleep(retry_interval)
            elapsed += retry_interval
            attempt += 1
    return False

def read_json(INPUT_FILE):
    with open(Path(INPUT_FILE), "r", encoding="utf-8") as f:
        data = json.load(f)
        
    pos = data.get("positive", None)
    neg = data.get("negative", None)
    out = data.get("output", "out_img")
    
    return (pos, neg, out)
    
def write_json(INPUT_FILE):
    pos, neg, out = read_json(INPUT_FILE)
    
    with open(Path("prompt_img.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    
    data["prompt"]["9"]["inputs"]["filename_prefix"] = str(INPUT_FILE).split('/')[-1].strip('.json')
    if pos: data["prompt"]["4"]["inputs"]["text"] = pos
    if neg: data["prompt"]["5"]["inputs"]["text"] = neg
    
    with open(Path("prompt_img.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    return out
    
def fetch_image(INPUT_FILE):
    OUTPUT_DIR = write_json(INPUT_FILE)
    TARGET_FILE = "prompt_img.json"
    
    cmd = f'curl -s -X POST -H "Content-Type: application/json" -d @{TARGET_FILE} http://127.0.0.1:8188/prompt | jq'
    subprocess.run(cmd, shell=True)

    print("[GOOD] 이미지 생성 요청 완료.")

    # 생성물 조회/다운로드 시에도 stem 사용
    file_name = str(INPUT_FILE).split('/')[-1].strip('.json')
    url = f"http://127.0.0.1:8188/view?filename={file_name}_00001_.png&type=output"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    download_image(url, str(Path(OUTPUT_DIR) / f"{file_name}.png"))
    return True

def request_queue(q_dir):
    q_path = Path(q_dir)
    files = sorted(q_path.glob("*.json"))
    if not files:
        print(f"[INFO] 큐가 비어있습니다: {q_path.resolve()}")
        return False
    
    for image in files: fetch_image(INPUT_FILE= image)

if __name__ == "__main__":
    request_queue("img_q")