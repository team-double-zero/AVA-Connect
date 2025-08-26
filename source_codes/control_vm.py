#!/usr/bin/env python3

import os
import re
import time
import socket
import shlex
import subprocess
import errno
import paramiko
from typing import Tuple, Dict, Any, List

from dotenv import load_dotenv
from vastai_sdk import VastAI

load_dotenv()

POLL_INTERVAL = 3     # 초
TIMEOUT_SEC = 300   # 최대 대기 5분

API_KEY = os.getenv("VAST_API_KEY")
INSTANCE_ID = int(os.getenv("VAST_INSTANCE_ID", "0"))
if not API_KEY or not INSTANCE_ID:
    raise SystemExit("VAST_API_KEY / VAST_INSTANCE_ID 를 .env 에 설정하세요.")

WAIT_BOOT_TIMEOUT = int(os.getenv("WAIT_BOOT_TIMEOUT", "600"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))
SSH_KEY_PATH = os.path.expanduser(os.getenv("SSH_KEY_PATH", "~/.ssh/id_rsa"))
SSH_USER_FALLBACK = os.getenv("SSH_USER", "root")

TUNNEL_ENABLE = os.getenv("SSH_TUNNEL_ENABLE", "true").lower() == "true"
TUNNEL_PORT = int(os.getenv("SSH_TUNNEL_PORT", "8080"))
TUNNEL_REMOTE = os.getenv("SSH_TUNNEL_REMOTE", "127.0.0.1") + ':' + str(TUNNEL_PORT)
AUTO_SSH_ENABLE = os.getenv("AUTO_SSH_ENABLE", "false").lower() == "true"

def vm_stop():
    api_key = os.getenv("VAST_API_KEY")
    instance_id = int(os.getenv("VAST_INSTANCE_ID", "0"))

    if not api_key or not instance_id:
        raise SystemExit("❌ 환경변수 VAST_API_KEY / VAST_INSTANCE_ID 확인 필요")

    api = VastAI(api_key=api_key)

    # 현재 상태 확인
    inst = api.show_instance(id=instance_id)   # ✅ 여기 수정
    cur = inst.get("cur_state")
    print(f"[INFO] Instance {instance_id} 현재 상태: {cur}")

    if cur == "stopped":
        print("[OK ] 이미 stopped 상태입니다.")
        return

    # 정지 요청
    print(f"[API] stop_instance({instance_id}) 호출")
    api.stop_instance(id=instance_id)   # ✅ 여기도 id로 전달

    # 폴링하여 stopped 될 때까지 대기
    start = time.time()
    while True:
        inst = api.show_instance(id=instance_id)   # ✅ 동일 수정
        cur = inst.get("cur_state")
        intended = inst.get("intended_status")
        print(f"[WAIT] cur_state={cur}, intended={intended}")

        if cur == "stopped":
            print("[DONE] 인스턴스가 stopped 상태가 되었습니다.")
            break

        if time.time() - start > TIMEOUT_SEC:
            raise SystemExit("⏱️ 타임아웃: 원하는 시간 안에 stopped 상태가 되지 않았습니다.")

        time.sleep(POLL_INTERVAL)

# ---------- helpers ----------
def status_str(inst: Dict[str, Any]) -> str:
    return (inst.get("actual_status") or inst.get("status_msg") or inst.get("status") or "").lower()


def get_instance(client: VastAI, iid: int) -> Dict[str, Any] | None:
    for it in client.show_instances():
        if int(it.get("id", -1)) == iid:
            return it
    return None


def extract_ssh(inst: Dict[str, Any]) -> Tuple[str, str, int, List[Tuple[str, str, str, int]]]:
    """
    SSH 접속 후보들을 생성(우선순위 포함)하고, 최우선 후보를 반환.
    return: (user, host, port, variants)  # variants = [(tag, user, host, port), ...]
    우선순위:
    1) ssh 문자열 파싱 (예: 'ssh -p 56484 root@202.79.96.144')
    2) public_ipaddr + docker port mapping ('22/tcp' -> HostPort)
    3) ssh_host + ssh_port
    """
    variants: List[Tuple[str, str, str, int]] = []

    # 1) ssh 문자열 파싱
    ssh_cmd = inst.get("ssh") or inst.get("ssh_url") or inst.get("ssh_cmd")
    if ssh_cmd:
        # 포트
        m_port = re.search(r"-p\s+(\d+)", ssh_cmd)
        # user@host
        m_uh = re.search(r"([A-Za-z0-9._-]+)@([A-Za-z0-9._-]+)", ssh_cmd)
        if m_port and m_uh:
            port = int(m_port.group(1))
            user = m_uh.group(1)
            host = m_uh.group(2)
            variants.append(("ssh_cmd", user, host, port))

    # 2) public_ipaddr + docker 22/tcp port mapping
    host2 = inst.get("public_ipaddr") or inst.get("public_ip") or inst.get("ip")
    ports = inst.get("ports") or {}
    hostport = None
    if isinstance(ports, dict) and "22/tcp" in ports and ports["22/tcp"]:
        try:
            hostport = int(ports["22/tcp"][0]["HostPort"])
        except Exception:
            hostport = None
    if host2 and hostport:
        variants.append(("public_ip+map", "root", str(host2), hostport))

    # 3) ssh_host + ssh_port (프록시 호스트가 나오는 경우가 있음: sshN.vast.ai)
    host3 = inst.get("ssh_host")
    port3 = inst.get("ssh_port") or inst.get("port_ssh")
    if host3 and port3:
        variants.append(("ssh_host+port", "root", str(host3), int(port3)))

    if not variants:
        raise RuntimeError("SSH 접속 후보를 만들 수 없습니다.")

    # 우선순위 정렬: ssh_cmd > public_ip+map > ssh_host+port
    variants.sort(key=lambda x: 0 if x[0] == "ssh_cmd" else (1 if x[0] == "public_ip+map" else 2))
    tag, user, host, port = variants[0]
    return user, host, port, variants


def try_tcp(host: str, port: int, banner: bool = True) -> Tuple[bool, str]:
    """
    TCP 접속 시도 + (가능하면) SSH 배너 수신
    """
    try:
        with socket.create_connection((host, port), timeout=5) as sock:
            msg = "connect ok"
            if banner:
                try:
                    sock.settimeout(2)
                    b = sock.recv(1024).decode(errors="ignore").strip()
                    if b:
                        msg += f", banner='{b[:80]}'"
                    else:
                        msg += ", banner=none"
                except Exception as e:
                    msg += f", banner_err={e}"
            return True, msg
    except OSError as e:
        return False, f"tcp_error={e}"


def wait_ssh_multi(candidates: List[Tuple[str, int]], timeout: int = 600, interval: int = 3) -> Tuple[str, int]:
    """
    여러 (host, port) 후보에 대해 순환하면서 SSH 준비될 때까지 대기.
    성공하면 (host, port) 반환.
    """
    t0 = time.time()
    while True:
        for host, port in candidates:
            ok, detail = try_tcp(host, port, banner=True)
            print(f"[WAIT] {host}:{port} -> {detail}")
            if ok:
                print(f"[OK ] SSH ready on {host}:{port}")
                return host, port
        if time.time() - t0 > timeout:
            raise TimeoutError(f"SSH가 {timeout}s 안에 준비되지 않음: {candidates}")
        time.sleep(interval)

def is_local_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """
    해당 로컬 포트가 비어있는지 확인. 사용중이면 False.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        return True
    except OSError as e:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass

def debug_ssh_tunnel_verbose(user: str, host: str, port: int, key_path: str, local_port: int, remote: str, timeout: int = 12) -> None:
    """
    ssh -vvv 로 포워딩을 시도하고 표준에러를 수집해 디버그용으로 출력.
    블로킹되면 timeout 초 뒤 강제 종료.
    """
    key_path = os.path.expanduser(key_path)
    cmd = [
        "ssh",
        "-vvv",
        "-i", key_path,
        "-o", "IdentitiesOnly=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ConnectTimeout=10",
        "-p", str(port),
        "-L", f"{local_port}:{remote}",
        "-N",
        f"{user}@{host}",
    ]
    print("[DEBUG] verbose SSH test:\n       " + " ".join(shlex.quote(x) for x in cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        print("[DEBUG] verbose ssh RETCODE:", res.returncode)
        if res.stdout.strip():
            print("[DEBUG] verbose ssh STDOUT:\n" + res.stdout)
        if res.stderr.strip():
            print("[DEBUG] verbose ssh STDERR:\n" + res.stderr)
    except subprocess.TimeoutExpired as te:
        print(f"[DEBUG] verbose ssh TIMEOUT after {timeout}s")


def wait_local_port(port: int, host: str = "127.0.0.1", timeout: int = 60, interval: float = 0.5) -> None:
    """
    로컬 포트가 열릴 때까지 대기. 실패 시 TimeoutError.
    """
    t0 = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"로컬 포트 {host}:{port} 가 {timeout}s 내에 열리지 않음. "
                                   f"원격 SSH 접근/인증, 또는 -L {port}:{host}:{port} 구성이 올바른지 확인하세요.")
            time.sleep(interval)


def ensure_ssh_agent_with_key(primary_key_path: str, extra_key_path: str | None = None) -> None:
    """
    Ensure an ssh-agent is running for this process and add SSH keys.
    This mirrors running:
        eval "$(ssh-agent -s)"
        ssh-add <keys>
    but persists agent env (SSH_AUTH_SOCK/SSH_AGENT_PID) into this Python process.
    """
    # 1) Check whether an agent is already usable by attempting 'ssh-add -l'
    def _ssh_add_list(env=None) -> subprocess.CompletedProcess:
        return subprocess.run(["ssh-add", "-l"], capture_output=True, text=True, env=env)

    env = os.environ.copy()
    listed = _ssh_add_list(env=env)
    need_agent = False
    if listed.returncode != 0 or "The agent has no identities" in (listed.stderr + listed.stdout):
        need_agent = True

    if need_agent:
        # 2) Start a new agent and capture its environment variables
        proc = subprocess.run(["ssh-agent", "-s"], capture_output=True, text=True)
        out = proc.stdout.strip() + "\n" + proc.stderr.strip()
        # Parse SSH_AUTH_SOCK and SSH_AGENT_PID
        m_sock = re.search(r"SSH_AUTH_SOCK=([^;]+);", out)
        m_pid = re.search(r"SSH_AGENT_PID=([0-9]+);", out)
        if not (m_sock and m_pid):
            print("[SSH] ssh-agent start failed (could not parse env); output was:")
            print(out)
            return
        env["SSH_AUTH_SOCK"] = m_sock.group(1)
        env["SSH_AGENT_PID"] = m_pid.group(1)
        # Persist into current process so subsequent calls inherit it
        os.environ["SSH_AUTH_SOCK"] = env["SSH_AUTH_SOCK"]
        os.environ["SSH_AGENT_PID"] = env["SSH_AGENT_PID"]
        print(f"[SSH] agent started (pid={env['SSH_AGENT_PID']})")

    # 3) Add keys (primary + extra)
    keys_to_add = []
    if primary_key_path:
        keys_to_add.append(os.path.expanduser(primary_key_path))
    if extra_key_path:
        keys_to_add.append(os.path.expanduser(extra_key_path))

    for kp in keys_to_add:
        try:
            if os.path.isfile(kp):
                res = subprocess.run(["ssh-add", kp], capture_output=True, text=True, env=os.environ.copy())
                if res.returncode == 0:
                    print(f"[SSH] ssh-add ok: {kp}")
                else:
                    # If key is already added, ssh-add prints to stderr; we show it once for visibility.
                    msg = (res.stdout + res.stderr).strip()
                    print(f"[SSH] ssh-add returncode={res.returncode} for {kp}: {msg}")
            else:
                print(f"[SSH] key file not found, skip: {kp}")
        except Exception as e:
            print(f"[SSH] ssh-add error for {kp}: {e}")


def open_ssh_tunnel(user: str, host: str, port: int, key_path: str, local_port: int, remote: str) -> int:
    """
    SSH 로컬 포트포워딩을 백그라운드(-fN)로 열고, 로컬 포트가 실제로 열릴 때까지 대기.
    반환값: ssh 프로세스 PID
    """
    key_path = os.path.expanduser(key_path)
    if not os.path.isfile(key_path):
        raise FileNotFoundError(f"SSH 키가 존재하지 않습니다: {key_path}")

    # 로컬 포트 사용중인지 선확인
    subprocess.run(f"lsof -ti:{TUNNEL_PORT} | xargs kill -9", shell=True)

    ensure_ssh_agent_with_key(key_path, os.path.expanduser("~/.ssh/id_rsa_vast"))

    cmd = [
        "ssh",
        "-i", key_path,
        "-o", "IdentitiesOnly=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ConnectTimeout=10",
        "-p", str(port),
        "-L", f"{local_port}:{remote}",
        "-fN",
        f"{user}@{host}",
    ]
    print("[TUNNEL] opening:", " ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # ssh가 백그라운드로 넘어가며 바로 종료(에러)했는지 짧게 확인
    time.sleep(0.4)
    if proc.poll() is not None and proc.returncode != 0:
        try:
            err_text = proc.stderr.read().decode(errors="ignore")
        except Exception:
            err_text = ""
        raise RuntimeError(f"SSH 터널 프로세스가 즉시 종료(returncode={proc.returncode}).\n{err_text}")

    # 포트가 실제로 열릴 때까지 대기
    try:
        wait_local_port(local_port, host="127.0.0.1", timeout=60, interval=0.5)
    except TimeoutError as te:
        # 실패 시 즉시 프로세스 상태와 stderr 출력
        try:
            rc = proc.poll()
            err_snip = proc.stderr.read().decode(errors='ignore')
            print(f"[DEBUG] ssh rc={rc}, stderr:\n{err_snip}")
        except Exception:
            pass
        raise

    print(f"[TUNNEL] ready on 127.0.0.1:{local_port} (pid={proc.pid})")
    return proc.pid


# ---------- main ----------
def vm_start():
    api = VastAI(api_key=API_KEY)

    # 상태 조회
    inst = get_instance(api, INSTANCE_ID)
    if not inst:
        raise SystemExit(f"인스턴스 {INSTANCE_ID} 를 찾을 수 없습니다.")
    st = status_str(inst)
    print(f"[INFO] VM 을 켭니다. 현재 상태: {st}")
    print(f"[DEBUG] WAIT_BOOT_TIMEOUT={WAIT_BOOT_TIMEOUT}, POLL_INTERVAL={POLL_INTERVAL}")

    # running 여부 판단
    is_running = ("running" in st) or (st in {"start", "started", "on"})
    if not is_running:
        print("[API] start_instance 호출")
        try:
            api.start_instance(id=INSTANCE_ID)
        except Exception as e:
            print(f"[WARN] start_instance: {e}")

        # running 대기
        t0 = time.time()
        while True:
            inst = get_instance(api, INSTANCE_ID) or {}
            st = status_str(inst)
            elapsed = int(time.time() - t0)
            print(f"[WAIT] status={st or 'unknown'} ({elapsed}s)")
            if ("running" in st) or (st in {"start", "started", "on"}):
                break
            if elapsed > WAIT_BOOT_TIMEOUT:
                raise TimeoutError(f"{WAIT_BOOT_TIMEOUT}s 내에 running 되지 않음 (마지막 상태: {st})")
            time.sleep(POLL_INTERVAL)
    else:
        print("[SKIP] 이미 running 상태 → 즉시 SSH 준비 진행")

    # SSH 대상/포트 후보 생성
    user, host, port, variants = extract_ssh(inst)
    if not user:
        user = SSH_USER_FALLBACK

    print("[DEBUG] SSH candidates (prio order):")
    for tag, u, h, p in variants:
        print(f"  - {tag}: {u}@{h}:{p}")

    # 후보들에 대해 준비 대기 (이미 running이면 곧바로 통과할 확률 높음)
    host, port = wait_ssh_multi([(h, p) for _, _, h, p in variants], timeout=WAIT_BOOT_TIMEOUT, interval=3)
    print(f"[INFO] SSH 대상 확정: {user}@{host}:{port}")

    # 터널 실제 개설(선택)
    tunnel_pid = None
    if TUNNEL_ENABLE:
        try:
            tunnel_pid = open_ssh_tunnel(
                user=user,
                host=host,
                port=port,
                key_path=SSH_KEY_PATH,
                local_port=TUNNEL_PORT,
                remote=TUNNEL_REMOTE,
            )
            print(f"[GOOD] 터널 개설 완료 → http://127.0.0.1:{TUNNEL_PORT}")
        except Exception as e:
            print(f"[ERR] 터널 개설 실패: {e}")
            print("      수동으로 아래 명령을 실행해보세요:")
            print("      ssh -i", shlex.quote(SSH_KEY_PATH), "-o", "IdentitiesOnly=yes", "-p", str(port),
                  "-L", f"{TUNNEL_PORT}:{TUNNEL_REMOTE}", f"{user}@{host}")
            print("[DEBUG] 원인 파악을 위해 -vvv 테스트를 수행합니다...")
            try:
                debug_ssh_tunnel_verbose(
                    user=user,
                    host=host,
                    port=port,
                    key_path=SSH_KEY_PATH,
                    local_port=TUNNEL_PORT,
                    remote=TUNNEL_REMOTE,
                    timeout=12,
                )
            except Exception as ee:
                print(f"[DEBUG] verbose 테스트 중 예외: {ee}")
    else:
        print("[INFO] SSH 터널 비활성화 상태(SSH_TUNNEL_ENABLE=false). 아래 명령으로 수동 개설 가능:")
        print("       ssh -i", shlex.quote(SSH_KEY_PATH), "-o", "IdentitiesOnly=yes", "-p", str(port),
              "-L", f"{TUNNEL_PORT}:{TUNNEL_REMOTE}", f"{user}@{host}")

    # AUTO SSH (interactive) if requested
    if AUTO_SSH_ENABLE:
        print("\n[AUTO] 인터랙티브 SSH 접속을 시작합니다.")
        ssh_args = [
            "ssh", "-i", SSH_KEY_PATH, "-o", "IdentitiesOnly=yes", "-p", str(port),
            f"{user}@{host}"
        ]
        # foreground interactive session
        subprocess.call(ssh_args)

    print("\n[DONE] 원격 VM 준비 완료!")
    
    return TUNNEL_ENABLE
