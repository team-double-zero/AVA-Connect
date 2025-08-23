import json
import subprocess
from pathlib import Path

import down     # 터널링된 url 로부터 파일 다운로드

PROMPT_VIDEO = Path("prompt_vid.json")


# 주어진 json 파일로부터 정보 가져오기
def read_json(INPUT_FILE):      # 파일 경로 -> (긍정 프롬, 부정 프롬, 다운로드 위치)
    p = Path(INPUT_FILE)
    with open(p, "r", encoding="utf-8") as f: data = json.load(f)

    pos = data.get("positive")
    neg = data.get("negative")
    out = data.get("output", p.stem)

    return (pos, neg, out)

# 불러온 정보로부터 prompt_().json 파일 작성 (모델 용) 
def write_json(INPUT_FILE):     # 파일 경로 -> 다운로드 위치
    FPS = 24
    LEN_SEC = 5
    
    pos, neg, out = read_json(INPUT_FILE)

    with open(PROMPT_VIDEO, "r", encoding="utf-8") as f: data = json.load(f)

    stem = Path(INPUT_FILE).stem

    data["prompt"]["97"]["inputs"]["image"] = stem+'.png'
    data["prompt"]["93"]["inputs"]["text"] = pos
    data["prompt"]["89"]["inputs"]["text"] = neg
    data["prompt"]["108"]["inputs"]["filename_prefix"] = stem
    data["prompt"]["94"]["inputs"]["fps"] = FPS
    data["prompt"]["98"]["inputs"]["length"] = FPS*LEN_SEC

    with open(PROMPT_VIDEO, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=4)

    return out


def send_to_VM(INPUT_FILE):
    p = Path(INPUT_FILE).stem
    # 로컬(out_img)에서 생성된 이미지를 ComfyUI input으로 업로드
    # 터널링이 열려 있으므로 127.0.0.1:8188 로 POST
    root = Path(__file__).resolve().parents[1]  # AVA-Connect 루트
    local_img = (root / "out_img" / f"{p}.png").resolve()

    if not local_img.is_file():
        raise FileNotFoundError(f"로컬 이미지가 없습니다: {local_img}")
    cmd = (
        f'curl -sS --fail -X POST '
        f'-F "image=@{local_img};filename={p}.png" '
        f'-F "type=input" -F "subfolder=" -F "overwrite=true" '
        f'http://127.0.0.1:8188/upload/image'
    )
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise RuntimeError(f"업로드 실패: {local_img}")


# 비디오 생성 요청 및 다운로드
def fetch_video(INPUT_FILE, debug= False):
    write_json(INPUT_FILE)     # 파일 읽기, 전달 파일 작성, 다운로드 위치 설정
    
    OUTPUT_DIR = "../out_vid"
    
    send_to_VM(INPUT_FILE)
    
    if not debug:
        cmd = f'curl -s -X POST -H "Content-Type: application/json" -d @{PROMPT_VIDEO} http://127.0.0.1:8188/prompt | jq'
        subprocess.run(cmd, shell=True)
        print("[GOOD] 비디오 생성 요청 완료.")

        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        down.download(INPUT_FILE, 'mp4', OUTPUT_DIR)
    
    return True


# Queue 디렉토리 안의 모든 json에 대해 반복
def request_queue(q_dir, debug= False):
    q_path = Path(q_dir)
    files = sorted(q_path.glob("*.json"))
    if not files:
        print(f"[INFO] 큐가 비어있습니다: {q_path.resolve()}")
        return False
    
    for video in files: 
        if debug: print(f"[DEBUG] {video}")
        fetch_video(INPUT_FILE= video, debug= debug)