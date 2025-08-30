import vast_helper
from queue_controller import QueueManager
from tunnel_manager import TunnelManager

def check_Connection(interval: int = 5, timeout: int = 300):
    import requests, time
    for _ in range(int(timeout/interval)):
        r = requests.get()
        if r.status_code == 200:
            return True
        time.sleep(interval)
    return False


"""이미 ComfyUI 돌아가고 있을때 예외처리 하자!!"""

def run_ComfyUI():
    host, port, isNew = vast_helper.run_best_instance()
    if host and port:
        print("[INFO] Starting Initialization")
        
        tunnelManager = TunnelManager(host, port)
        cmd = (
            "set -euo pipefail; "
            "if [ -d ComfyUI ]; then cd ComfyUI && git pull; else git clone https://github.com/ahnjh05141/ComfyUI && cd ComfyUI; fi && "
            "chmod +x ava-initialize.sh && "
            "bash ./ava-initialize.sh && "
            "nohup python3 main.py --listen 0.0.0.0 --port 8080 > comfyui.log 2>&1 &"
        )
        tunnelManager.run_ssh_command(cmd)
        
        print("[INFO] Waiting for connecting ComfyUI")
        
        return check_Connection()
    
    
def main():
    connected = run_ComfyUI()
    if connected:
        qm = QueueManager()


if __name__ == "__main__":
    main()