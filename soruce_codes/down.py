import time, requests
from pathlib import Path
from urllib.parse import urlparse, parse_qs

def download(INPUT_FILE: str, FILE_TYPE: str, retry_interval: int = 3, max_wait: int = 600, debug= False):
    elapsed = 0
    attempt = 1
    pp = '../out_img' if FILE_TYPE =='png' else '../out_vid'
    file_name = Path(INPUT_FILE).stem
    sp = Path(f"{pp}/{file_name}.{FILE_TYPE}")
    
    url = f"http://127.0.0.1:8188/view?filename={file_name}_00001_.{FILE_TYPE}&type=output"
    
    print(f"[INFO] {max_wait}초 동안 다운로드를 시도합니다.")
    print(f"[INFO] {url}")

    if debug: print(INPUT_FILE, sp, url); return False
    else:
        if sp.exists() and sp.is_dir():
            try:
                q = parse_qs(urlparse(url).query)
                fname = q.get("filename", [INPUT_FILE])[0]
            except Exception:
                fname = INPUT_FILE
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