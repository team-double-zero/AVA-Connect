import os
from dotenv import load_dotenv
from vastai_sdk import VastAI
from collections import deque

class GpuManager():
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GpuManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        load_dotenv()
        self.API_KEY = os.getenv("VAST_API_KEY")
        self.INSTANCE_ID = os.getenv("VAST_INSTANCE_ID")
        self.client = VastAI(api_key=self.API_KEY)
        
        self.wait_queue = deque([])
    
    def add_queue(self, content_type: str, file_dir: str):
        new_item = (content_type, file_dir)
        self.wait_queue.append()

if __name__ == "__main__":
    gpuManager = GpuManager()