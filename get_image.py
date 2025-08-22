import json, requests
import time
import subprocess
import sys, os, re
from pathlib import Path

def download_image(url: str, save_path: str, retry_interval: int = 3, max_wait: int = 100):
    elapsed = 0
    attempt = 1
    print(f"[시도 {url}")

    while elapsed < max_wait:
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()

            with open(save_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)

            print(f"[COMPLETE] {save_path}에 다운로드 완료")
            return True

        except Exception as e:
            if elapsed + retry_interval >= max_wait:
                print("⏹ 최대 대기 시간 초과. 다운로드 실패.")
                return False
            print(f"loading page... {attempt}")
            time.sleep(retry_interval)
            elapsed += retry_interval
            attempt += 1
    return False

def read_json(INPUT_FILE):
    with open(Path(INPUT_FILE), "r", encoding="utf-8") as f:
        data = json.load(f)
        
    pos = data.get("positive", None)
    neg = data.get("negative", None)
    out = data.get("output", "img_out")
    
    return (pos, neg, out)
    
def write_json(INPUT_FILE):
    pos, neg, out = read_json(INPUT_FILE)
    
    with open(Path("image_prompt.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    
    data["prompt"]["9"]["inputs"]["filename_prefix"] = str(INPUT_FILE).split('/')[-1].strip('.json')
    if pos: data["prompt"]["4"]["inputs"]["text"] = pos
    if neg: data["prompt"]["5"]["inputs"]["text"] = neg
    
    with open(Path("image_prompt.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    return out
    
def fetch_image(INPUT_FILE):
    OUTPUT_DIR = write_json(INPUT_FILE)
    TARGET_FILE = "image_prompt.json"
    
    cmd = f'curl -s -X POST -H "Content-Type: application/json" -d @{TARGET_FILE} http://127.0.0.1:8188/prompt | jq'
    subprocess.run(cmd, shell=True)

    time.sleep(10)

    # 생성물 조회/다운로드 시에도 stem 사용
    url = f"http://127.0.0.1:8188/view?filename={INPUT_FILE}_00001_.png&type=output"
    download_image(url, f"{OUTPUT_DIR}")
    return True

def request_queue(q_dir):
    q_path = Path(q_dir)
    files = sorted(q_path.glob("*.json"))
    if not files:
        print(f"[INFO] 대상 폴더에 JSON이 없습니다: {q_path.resolve()}")
        return False
    
    failed = 0
    for image in files:
        try:
            ok = fetch_image(INPUT_FILE= image)
            if not ok:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] 처리 실패: {image} ({e})")
    total = len(files)
    print(f"[SUMMARY] {len(files)-failed} Success out of {len(files)}")
    return failed == 0

if __name__ == "__main__":
    request_queue("img_q")