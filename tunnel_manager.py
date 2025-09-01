import os
import subprocess
import paramiko
from dotenv import load_dotenv

class TunnelManager():
    load_dotenv()
    SSH_KEY_PATH = os.getenv("SSH_KEY_PATH") # 절대 경로
    
    def __init__(self, host, port):
        """host: ssh.vast.ai / local_port: 8080 / tunnel_port: 5자리수 동적 포트 """
        self.host = host
        self.local_port = 8090
        self.tunnel_port = port
        
        # ssh 키 재인식
        subprocess.run(f"ssh-add {self.SSH_KEY_PATH}", shell= True)
        
        # local_port (8080) 비우기
        subprocess.run(f"lsof -ti:{self.local_port} | xargs kill -9", shell=True)
        
        # 터널링 연결
        subprocess.run(f"ssh -i {self.SSH_KEY_PATH} -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p {self.tunnel_port} -N -f -L {self.local_port}:localhost:{8080} root@{self.host}", shell= True)


    def run_ssh_command(self, command: str):
        """원격 SSH 서버에서 명령어 실행하고 출력 및 에러 (stdout, stderr) 문자열 출력"""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 호스트 키 자동 승인
        ssh.connect(
            hostname= self.host,
            port= self.tunnel_port,
            username= "root",
            key_filename= self.SSH_KEY_PATH,
            look_for_keys= False
        )

        _, stdout, stderr = ssh.exec_command(command)

        for line in iter(stdout.readline, ""):
            print(line, end="")

        for line in iter(stderr.readline, ""):
            print(line, end="")

        ssh.close()


    def check_comfyui_connection(self, interval: int = 5, timeout: int = 300) -> bool:
        """
        터널링된 로컬 포트(예: 127.0.0.1:{self.local_port})로 ComfyUI HTTP 응답을 기다립니다.
        - TCP 레벨에서 먼저 포트가 열렸는지 확인
        - 이어서 HTTP(HEAD → 실패 시 GET)로 2xx~4xx 응답을 '응답 있음'으로 간주
        - 성공 시 True, 타임아웃 시 False 반환
        """
        import time
        import socket
        import requests
        
        url = f"http://127.0.0.1:{self.local_port}/"
        interval = max(1, int(interval))
        timeout = max(1, int(timeout))
        deadline = time.monotonic() + timeout
        attempt = 0
        
        print(f"[INFO] Waiting for ComfyUI on {url} (timeout={timeout}s, interval={interval}s)")
        
        while time.monotonic() < deadline:
            attempt += 1
            
            # 1) TCP 포트 오픈 여부 확인 (빠른 실패/성공 판별)
            try:
                with socket.create_connection(("127.0.0.1", self.local_port), timeout=2):
                    tcp_ok = True
            except OSError as e:
                tcp_ok = False
                print(f"[WAIT] attempt {attempt}: TCP not ready ({e})")
            
            if not tcp_ok:
                time.sleep(interval)
                continue
            
            # 2) HTTP 응답 확인 (HEAD 우선, 실패 시 GET)
            try:
                resp = requests.head(url, timeout=5)
                if 200 <= resp.status_code < 500:
                    print(f"[OK ] ComfyUI responded (HEAD {resp.status_code})")
                    return True
            except requests.RequestException:
                # 일부 서버는 HEAD 미지원 → GET으로 재시도
                try:
                    resp = requests.get(url, timeout=8)
                    if 200 <= resp.status_code < 500:
                        print(f"[OK ] ComfyUI responded (GET {resp.status_code})")
                        return True
                    else:
                        print(f"[WAIT] attempt {attempt}: HTTP {resp.status_code}")
                except requests.RequestException as e:
                    print(f"[WAIT] attempt {attempt}: HTTP error ({e})")
            
            time.sleep(interval)
        
        print("[FAIL] Timed out waiting for ComfyUI.")
        return False


    """이미 ComfyUI 돌아가고 있을때 예외처리 하자!!"""

    def run_comfyui(self):
        """터널링된 ssh 상에서 ComfyUI 를 부팅합니다."""
        print("[INFO] 터널링된 ssh 상에서 ComfyUI 를 부팅합니다.")
        cmd = (
            "set -euo pipefail; "
            "if [ -d ComfyUI ]; then cd ComfyUI && git pull; else git clone https://github.com/ahnjh05141/ComfyUI && cd ComfyUI; fi && "
            "chmod +x ava-initialize.sh && "
            "bash ./ava-initialize.sh && "
            f"nohup python3 main.py --listen 0.0.0.0 --port {self.local_port} > comfyui.log 2>&1 &"
        )
        self.run_ssh_command(cmd)