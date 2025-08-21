#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vast.ai VM: 이미 running이면 즉시 SSH 헬스체크, 아니면 start→대기 후 체크

.env 예시:
  VAST_API_KEY=...
  VAST_INSTANCE_ID=25154352

  SSH_USER=root
  SSH_KEY_PATH=~/.ssh/id_rsa_vast

  WAIT_BOOT_TIMEOUT=600
  POLL_INTERVAL=5

  # (옵션) 터널 안내만 출력
  SSH_TUNNEL_ENABLE=false
  SSH_TUNNEL_LOCAL=8188
  SSH_TUNNEL_REMOTE=127.0.0.1:8188
"""

import os
import re
import time
import socket
import shlex
from typing import Tuple, Dict, Any, List

from dotenv import load_dotenv
import paramiko
from vastai_sdk import VastAI


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


def run_remote_command(host: str, port: int, user: str, key_path: str, command: str) -> tuple[str, str, int]:
    """
    Paramiko로 SSH 접속 후 임의 command 실행
    return: (stdout, stderr, exit_status)
    """
    key_path = os.path.expanduser(key_path)
    key = paramiko.RSAKey.from_private_key_file(key_path)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[SSH] connect {user}@{host}:{port}")
    ssh.connect(host, port=port, username=user, pkey=key)

    stdin, stdout, stderr = ssh.exec_command(command)
    out = stdout.read().decode(errors="ignore")
    err = stderr.read().decode(errors="ignore")
    status = stdout.channel.recv_exit_status()
    ssh.close()
    return out, err, status


# ---------- main ----------
def main():
    load_dotenv()

    API_KEY = os.getenv("VAST_API_KEY")
    INSTANCE_ID = int(os.getenv("VAST_INSTANCE_ID", "0"))
    if not API_KEY or not INSTANCE_ID:
        raise SystemExit("VAST_API_KEY / VAST_INSTANCE_ID 를 .env 에 설정하세요.")

    WAIT_BOOT_TIMEOUT = int(os.getenv("WAIT_BOOT_TIMEOUT", "600"))
    POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))
    SSH_KEY_PATH = os.path.expanduser(os.getenv("SSH_KEY_PATH", "~/.ssh/id_rsa"))
    SSH_USER_FALLBACK = os.getenv("SSH_USER", "root")

    TUNNEL_ENABLE = os.getenv("SSH_TUNNEL_ENABLE", "false").lower() == "true"
    TUNNEL_LOCAL = int(os.getenv("SSH_TUNNEL_LOCAL", "8188"))
    TUNNEL_REMOTE = os.getenv("SSH_TUNNEL_REMOTE", "127.0.0.1:8188")

    api = VastAI(api_key=API_KEY)

    # 상태 조회
    inst = get_instance(api, INSTANCE_ID)
    if not inst:
        raise SystemExit(f"인스턴스 {INSTANCE_ID} 를 찾을 수 없습니다.")
    st = status_str(inst)
    print(f"[INFO] 현재 상태: {st}")
    print(f"[DEBUG] WAIT_BOOT_TIMEOUT={WAIT_BOOT_TIMEOUT}, POLL_INTERVAL={POLL_INTERVAL}")

    # running 여부 판단
    is_running = ("running" in st) or (st in {"start", "started", "on"})
    if not is_running:
        print("[API ] start_instance 호출")
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

    # 터널 명령 안내(선택)
    if TUNNEL_ENABLE:
        tunnel_cmd = [
            "ssh", "-i", SSH_KEY_PATH, "-o", "IdentitiesOnly=yes", "-p", str(port),
            "-L", f"{TUNNEL_LOCAL}:{TUNNEL_REMOTE}", f"{user}@{host}"
        ]
        print("[TUNNEL] 로컬 포트포워딩 명령:")
        print("         " + " ".join(shlex.quote(x) for x in tunnel_cmd))

    # 기본 헬스체크 (ping, HTTP, GPU)
    print("\n[CHECK] 네트워크/시스템 스모크 테스트")
    out, err, code = run_remote_command(host, port, user, SSH_KEY_PATH, "ping -c 4 8.8.8.8")
    print("[PING out]\n", out)
    if err.strip():
        print("[PING err]\n", err)
    print("[PING code]", code)

    out, err, code = run_remote_command(
        host, port, user, SSH_KEY_PATH,
        "which curl >/dev/null 2>&1 && curl -I https://www.google.com || wget -S --spider https://www.google.com"
    )
    print("[HTTP out]\n", out)
    if err.strip():
        print("[HTTP err]\n", err)
    print("[HTTP code]", code)

    out, err, code = run_remote_command(host, port, user, SSH_KEY_PATH, "nvidia-smi || echo 'no nvidia-smi'")
    print("[GPU out]\n", out)
    if err.strip():
        print("[GPU err]\n", err)
    print("[GPU code]", code)

    # 수동 접속 명령 안내
    ssh_cmd = [
        "ssh", "-i", SSH_KEY_PATH, "-o", "IdentitiesOnly=yes", "-p", str(port),
        f"{user}@{host}"
    ]
    print("\n[READY] 수동 접속 명령:")
    print("       " + " ".join(shlex.quote(x) for x in ssh_cmd))
    print("\n[DONE] 원격 VM 준비 완료!")


if __name__ == "__main__":
    main()