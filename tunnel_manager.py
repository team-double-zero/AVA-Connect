import os
import subprocess
import paramiko
from dotenv import load_dotenv

class TunnelManager():
    load_dotenv()
    SSH_KEY_PATH = os.getenv("SSH_KEY_PATH")
    PARAMIKO_KEY_PATH = "/Users/ahn/.ssh/id_rsa_vast"
    
    def __init__(self, host, port):
        """host: ssh.vast.ai / local_port: 8080 / tunnel_port: 5자리수 동적 포트 """
        self.host = host
        self.local_port = 8080
        self.tunnel_port = port
        
        # ssh 키 재인식
        subprocess.run(f"ssh-add {self.SSH_KEY_PATH}", shell= True)
        
        # local_port (8080) 비우기
        subprocess.run(f"lsof -ti:{self.local_port} | xargs kill -9", shell=True)
        
        # 터널링 연결
        subprocess.run(f"ssh -i {self.SSH_KEY_PATH} -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p {self.tunnel_port} -N -f -L {self.local_port}:localhost:{self.local_port} root@{self.host}", shell= True)


    def run_ssh_command(self, command: str):
        """원격 SSH 서버에서 명령어 실행하고 출력 및 에러 (stdout, stderr) 문자열 출력"""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 호스트 키 자동 승인
        ssh.connect(
            hostname= self.host,
            port= self.tunnel_port,
            username= "root",
            key_filename= self.PARAMIKO_KEY_PATH,
            look_for_keys= False
        )

        stdin, stdout, stderr = ssh.exec_command(command)

        print("[DEBUG] ssh command (stdout)")
        for line in iter(stdout.readline, ""):
            print(line, end="")

        print("[DEBUG] ssh command (stderr)")
        for line in iter(stderr.readline, ""):
            print(line, end="")

        ssh.close()

        return stdout, stderr
