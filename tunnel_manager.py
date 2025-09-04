import os
import time
import subprocess
import paramiko
from dotenv import load_dotenv

class TunnelManager():
    load_dotenv()
    ssh_key_path:str = None # 절대 경로
    
    def __init__(self, host, tunnel_port, local_port=8090, ssh_key_path=None):
        """host: ssh.vast.ai / local_port: 8090 / tunnel_port: 5자리수 동적 포트 """
        self.host = host
        self.tunnel_port = tunnel_port
        self.local_port = local_port
        self.ssh_key_path = ssh_key_path or os.getenv("SSH_KEY_PATH")
        
        # ssh 키 재인식
        subprocess.run(f"ssh-add {self.ssh_key_path}", shell= True)
        
        # local_port (8080) 비우기
        subprocess.run(f"lsof -ti:{self.local_port} | xargs kill -9", shell=True)
        
        # 터널링 연결
        subprocess.run(f"ssh -i {self.ssh_key_path} -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p {self.tunnel_port} -N -f -L {self.local_port}:localhost:{8080} root@{self.host}", shell= True)


    def run_ssh_command(self, command: str):
        """원격 SSH 서버에서 명령어 실행하고 출력 및 에러 (stdout, stderr) 문자열 출력"""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 호스트 키 자동 승인
        ssh.connect(
            hostname= self.host,
            port= self.tunnel_port,
            username= "root",
            key_filename= self.ssh_key_path,
            look_for_keys= False
        )

        _, stdout, stderr = ssh.exec_command(command)

        # 출력 수집
        stdout_content = ""
        stderr_content = ""
        
        for line in iter(stdout.readline, ""):
            print(line, end="")
            stdout_content += line

        for line in iter(stderr.readline, ""):
            print(line, end="")
            stderr_content += line

        # Exit code 확인
        exit_code = stdout.channel.recv_exit_status()
        
        ssh.close()
        
        # 명령어 실패 시 예외 발생
        if exit_code != 0:
            error_msg = f"SSH 명령어 실패 (exit code: {exit_code})"
            if stderr_content.strip():
                error_msg += f"\nSTDERR: {stderr_content.strip()}"
            raise RuntimeError(error_msg)


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
        """견고한 ComfyUI 부팅"""
        print("[INFO] 터널링된 ssh 상에서 ComfyUI 를 부팅합니다.")
        
        # 단계별 실행으로 오류 지점 파악 가능하도록 변경
        commands = [
            self._get_setup_command(),
            self._get_installation_command(), 
            self._get_startup_command()
        ]
        
        for i, cmd in enumerate(commands, 1):
            try:
                print(f"[STEP {i}/3] {['환경 설정', '종속성 설치', 'ComfyUI 시작'][i-1]}")
                self.run_ssh_command(cmd)
                time.sleep(2)  # 각 단계 간 잠시 대기
            except Exception as e:
                print(f"[ERROR] Step {i} 실패: {e}")
                if i == 1:  # 환경 설정 실패 시 복구 시도
                    self._attempt_setup_recovery()
                elif i == 2:  # 설치 단계 실패 시 복구 시도
                    self._attempt_installation_recovery()
                else:
                    raise

    def _get_setup_command(self) -> str:
        """환경 설정 명령어 (git clone 실패 시 재시도 포함)"""
        return """
        set -e;
        echo "[SETUP] 환경 준비 중...";
        
        # 임시 디렉터리 보장 및 권한 설정 (pip/temp용)
        mkdir -p /tmp /var/tmp /usr/tmp /root/tmp || true;
        chmod 1777 /tmp /var/tmp /usr/tmp || true;
        chmod 777 /root/tmp || true;
        
        # 기본 캐시 디렉터리 정리로 공간 확보
        rm -rf ~/.cache/pip || true;
        rm -rf ~/.cache/huggingface || true;
        
        df -h || true;
        
        # 기존 ComfyUI 프로세스 종료
        pkill -f 'python.*main.py' || true;
        
        # ComfyUI 디렉토리 준비 (실패 시 재시도)
        if [ -d ComfyUI ]; then 
            echo "[SETUP] 기존 ComfyUI 발견, 업데이트 중...";
            cd ComfyUI && git pull --no-edit || true;
        else 
            echo "[SETUP] 사용자 ComfyUI 포크 클론 시도 1...";
            if ! git clone https://github.com/ahnjh05141/ComfyUI.git; then
                echo "[SETUP] 첫 번째 시도 실패, 재시도 중...";
                sleep 5;
                if ! git clone https://github.com/ahnjh05141/ComfyUI.git; then
                    echo "[SETUP] git clone 재시도 실패, wget으로 다운로드 시도...";
                    rm -rf ComfyUI* || true;
                    wget -O comfyui.zip https://github.com/ahnjh05141/ComfyUI/archive/refs/heads/main.zip || {
                        echo "[ERROR] 모든 다운로드 방법 실패";
                        exit 1;
                    };
                    unzip -q comfyui.zip && mv ComfyUI-main ComfyUI && rm comfyui.zip;
                    echo "[SETUP] wget으로 다운로드 완료";
                fi;
            fi;
        fi;
        
        # ComfyUI 폴더 존재 확인
        if [ ! -d ComfyUI ]; then
            echo "[ERROR] ComfyUI 폴더가 생성되지 않았습니다";
            exit 1;
        fi;
        
        echo "[SETUP] 환경 준비 완료 - ComfyUI 폴더 확인됨";
        """

    def _get_installation_command(self) -> str:
        """설치 명령어 (오류 허용)"""
        return f"""
        set -e;
        cd ComfyUI;
        echo "[INSTALL] 종속성 설치 시작...";
        
        # 임시/캐시 디렉터리 설정 (공간 부족/권한 이슈 완화)
        mkdir -p /root/tmp /root/.cache/pip /root/.cache/huggingface || true;
        chmod 777 /root/tmp || true;
        export TMPDIR=/root/tmp;
        export PIP_CACHE_DIR=/root/.cache/pip;
        export HF_HOME=/root/.cache/huggingface;
        
        df -h || true;
        
        # pip 업그레이드 시도 (실패해도 계속)
        python3 -m pip install --upgrade pip --user --no-cache-dir || {{
            echo "[WARN] pip 업그레이드 실패, 기존 pip 사용";
        }};
        
        # ava-initialize.sh가 있는지 확인
        if [ -f "ava-initialize.sh" ]; then
            echo "[INSTALL] ava-initialize.sh 발견 - 사용자 포크 설정으로 설치";
            chmod +x ava-initialize.sh;
            timeout 1800 env TMPDIR=/root/tmp HF_HOME=/root/.cache/huggingface bash ./ava-initialize.sh || {{
                echo "[WARN] ava-initialize.sh 실패, 기본 설치로 폴백";
                python3 -m pip install torch torchvision --user --no-cache-dir || echo "[WARN] torch 설치 실패";
                python3 -m pip install -r requirements.txt --user --no-cache-dir || echo "[WARN] requirements 설치 실패";
            }};
        else
            echo "[INSTALL] ava-initialize.sh 없음 - 기본 설치 진행";
            python3 -m pip install torch torchvision --user --no-cache-dir || echo "[WARN] torch 설치 실패";
            python3 -m pip install -r requirements.txt --user --no-cache-dir || echo "[WARN] requirements 설치 실패";
        fi;
        
        # 설치 후 캐시 정리로 공간 회수
        python3 -m pip cache purge || true;
        rm -rf ~/.cache/pip ~/.cache/huggingface || true;
        df -h || true;
        
        echo "[INSTALL] 설치 완료 (일부 오류 무시됨)";
        """

    def _get_startup_command(self) -> str:
        """시작 명령어"""
        return f"""
        set -e;
        cd ComfyUI;
        echo "[START] ComfyUI 시작 중...";
        
        # 로그 디렉토리 생성 (디스크 부족 시도 대비 전 처리)
        mkdir -p logs || true;
        # 불필요 캐시 추가 정리
        rm -rf ~/.cache/pip ~/.cache/huggingface || true;
        df -h || true;
        
        # ComfyUI 백그라운드 실행
        nohup python3 main.py --listen 0.0.0.0 --port 8080 > logs/comfyui.log 2>&1 &
        
        # 프로세스 ID 저장
        echo $! > logs/comfyui.pid;
        
        echo "[START] ComfyUI 시작 완료 (PID: $(cat logs/comfyui.pid))";
        sleep 3;
        """

    def _attempt_setup_recovery(self):
        """환경 설정 실패 시 ComfyUI 다운로드 재시도"""
        print("[RECOVERY] ComfyUI 다운로드 복구 시도 중...")
        
        recovery_cmd = """
        set -e;
        echo "[RECOVERY] ComfyUI 다운로드 복구 모드";
        
        # 기존 실패한 파일들 정리
        rm -rf ComfyUI* comfyui* || true;
        
        # 더 강력한 다운로드 시도 - 사용자 포크를 우선으로
        echo "[RECOVERY] 사용자 ComfyUI 포크 저장소 클론 시도...";
        if git clone https://github.com/ahnjh05141/ComfyUI.git; then
            echo "[RECOVERY] 사용자 포크 저장소 클론 성공";
        else
            echo "[RECOVERY] 백업 방법 시도...";
            # 백업 설치 방법들
            if git clone https://github.com/comfyanonymous/ComfyUI.git; then
                echo "[RECOVERY] 원본 저장소 클론 성공 (백업)";
                echo "[WARNING] 원본 저장소 사용 - ava-initialize.sh 없음";
            elif wget --no-check-certificate -O comfyui.tar.gz https://github.com/ahnjh05141/ComfyUI/archive/main.tar.gz && tar -xzf comfyui.tar.gz && mv ComfyUI-main ComfyUI && rm comfyui.tar.gz; then
                echo "[RECOVERY] 사용자 포크 tar.gz 다운로드 성공";
            else
                echo "[ERROR] 모든 복구 시도 실패";
                exit 1;
            fi;
        fi;
        
        # 최종 확인
        if [ ! -d ComfyUI ]; then
            echo "[ERROR] 복구 후에도 ComfyUI 폴더가 없습니다";
            exit 1;
        fi;
        
        echo "[RECOVERY] ComfyUI 복구 완료";
        """
        
        try:
            self.run_ssh_command(recovery_cmd)
            print("[RECOVERY] 환경 설정 복구 성공")
        except Exception as e:
            print(f"[RECOVERY] 환경 설정 복구 실패: {e}")
            raise

    def _attempt_installation_recovery(self):
        """설치 실패 시 복구 시도"""
        print("[RECOVERY] 설치 복구 시도 중...")
        
        recovery_cmd = """
        set -e;
        cd ComfyUI;
        echo "[RECOVERY] 복구 모드로 재설치 중...";
        
        # 기본 패키지만 설치 시도
        python3 -m pip install --user torch torchvision torchaudio || {{
            echo "[RECOVERY] torch 설치 실패, apt로 시도";
            sudo apt-get update;
            sudo apt-get install -y python3-torch python3-torchvision || true;
        }};
        
        # 필수 패키지만 설치
        python3 -m pip install --user pillow numpy requests || true;
        
        echo "[RECOVERY] 복구 완료";
        """
        
        try:
            self.run_ssh_command(recovery_cmd)
            print("[RECOVERY] 복구 성공")
        except Exception as e:
            print(f"[RECOVERY] 복구 실패: {e}")
            # 복구도 실패하면 기본 Python으로라도 시작 시도
            self._minimal_startup()

    def _minimal_startup(self):
        """최소한의 설정으로 ComfyUI 시작 시도"""
        print("[MINIMAL] 최소 설정으로 시작 시도...")
        
        minimal_cmd = """
        cd ComfyUI || exit 1;
        echo "[MINIMAL] 기본 설정으로 ComfyUI 시작...";
        
        # 최소한의 환경 설정
        export PYTHONPATH="${PYTHONPATH}:.";
        export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512";
        
        # 로그 디렉토리 생성
        mkdir -p logs;
        
        # 안전 모드로 ComfyUI 시작 (낮은 메모리 사용)
        nohup python3 main.py --listen 0.0.0.0 --port 8080 --lowvram --cpu > logs/minimal.log 2>&1 &
        
        echo $! > logs/minimal.pid;
        echo "[MINIMAL] 최소 모드 시작 완료";
        """
        
        try:
            self.run_ssh_command(minimal_cmd)
            print("[MINIMAL] 최소 모드 시작 성공")
        except Exception as e:
            print(f"[MINIMAL] 최소 모드도 실패: {e}")
            raise RuntimeError("모든 시작 방법 실패")

    def check_installation_progress(self) -> dict:
        """ComfyUI 설치 진행 상황 확인"""
        try:
            check_cmd = """
            echo "=== INSTALLATION STATUS ===";
            
            # ComfyUI 디렉토리 확인
            if [ -d "ComfyUI" ]; then
                echo "✅ ComfyUI 디렉토리 존재";
                cd ComfyUI;
                
                # 주요 파일들 확인
                if [ -f "main.py" ]; then echo "✅ main.py 존재"; else echo "❌ main.py 없음"; fi;
                if [ -f "requirements.txt" ]; then echo "✅ requirements.txt 존재"; else echo "❌ requirements.txt 없음"; fi;
                
                # 실행 중인 프로세스 확인
                if pgrep -f "python.*main.py" > /dev/null; then
                    echo "✅ ComfyUI 실행 중 (PID: $(pgrep -f 'python.*main.py'))";
                else
                    echo "❌ ComfyUI 실행되지 않음";
                fi;
                
                # 로그 확인
                if [ -f "logs/comfyui.log" ]; then
                    echo "📋 최근 로그 (마지막 5줄):";
                    tail -5 logs/comfyui.log 2>/dev/null || echo "로그 읽기 실패";
                fi;
                
            else
                echo "❌ ComfyUI 디렉토리 없음";
            fi;
            
            echo "=== END STATUS ===";
            """
            
            print("[CHECK] 설치 상태 확인 중...")
            self.run_ssh_command(check_cmd)
            
        except Exception as e:
            print(f"[CHECK] 상태 확인 실패: {e}")
            return {"status": "check_failed", "error": str(e)}

    def get_comfyui_logs(self, lines: int = 20) -> str:
        """ComfyUI 로그 가져오기"""
        try:
            log_cmd = f"""
            cd ComfyUI;
            if [ -f "logs/comfyui.log" ]; then
                echo "=== ComfyUI 로그 (최근 {lines}줄) ===";
                tail -{lines} logs/comfyui.log;
            elif [ -f "logs/minimal.log" ]; then
                echo "=== 최소 모드 로그 (최근 {lines}줄) ===";
                tail -{lines} logs/minimal.log;
            else
                echo "로그 파일을 찾을 수 없음";
            fi;
            """
            
            self.run_ssh_command(log_cmd)
            
        except Exception as e:
            print(f"[LOG] 로그 조회 실패: {e}")