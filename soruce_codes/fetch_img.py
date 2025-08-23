import json
import subprocess
from pathlib import Path

import down     # 터널링된 url 로부터 파일 다운로드

PROMPT_IMAGE = Path("prompt_img.json")


# 주어진 json 파일로부터 정보 가져오기
def read_json(INPUT_FILE):      # 파일 경로 -> (긍정 프롬, 부정 프롬, 다운로드 위치)
    p = Path(INPUT_FILE)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    pos = data.get("positive", None)
    neg = data.get("negative", None)
    out = data.get("output", p.stem)
    
    return (pos, neg, out)


# 불러온 정보로부터 prompt_().json 파일 작성 (모델 용) 
def write_json(INPUT_FILE):     # 파일 경로 -> 다운로드 위치
    pos, neg, out = read_json(INPUT_FILE)
    
    with open(PROMPT_IMAGE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    data["prompt"]["9"]["inputs"]["filename_prefix"] = Path(INPUT_FILE).stem
    if pos: data["prompt"]["4"]["inputs"]["text"] = pos
    if neg: data["prompt"]["5"]["inputs"]["text"] = neg
    
    with open(PROMPT_IMAGE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    return out


# 비디오 생성 요청 및 다운로드
def fetch_image(INPUT_FILE, debug= False):
    OUTPUT_DIR = write_json(INPUT_FILE)     # 파일 읽기, 전달 파일 작성, 다운로드 위치 설정
    
    if not debug:
        cmd = f'curl -s -X POST -H "Content-Type: application/json" -d @{PROMPT_IMAGE} http://127.0.0.1:8188/prompt | jq'
        subprocess.run(cmd, shell=True)
        print("[GOOD] 이미지 생성 요청 완료.")

    down.download(INPUT_FILE, 'png', debug= debug)
    
    return True


# Queue 디렉토리 안의 모든 json에 대해 반복
def request_queue(q_dir, debug= False):
    q_path = Path(q_dir)
    files = sorted(q_path.glob("*.json"))
    if not files:
        print(f"[INFO] 큐가 비어있습니다: {q_path.resolve()}")
        return False
    
    for image in files: 
        if debug: print(f"[DEBUG] {image}")
        fetch_image(INPUT_FILE= image, debug= debug)