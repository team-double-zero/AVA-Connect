import os
from dotenv import load_dotenv
from vastai_sdk import VastAI

load_dotenv()
API_KEY = os.getenv("VAST_API_KEY")
api = VastAI(api_key=API_KEY)

offers = api.search_offers(
    query="gpu_name~A100 rentable=true verified=true",
    limit=200
)

want = [o for o in offers if "gpu_name" in o and "a100" in o["gpu_name"].lower() and o.get("gpu_ram", 0) >= 40960]
want = sorted(want, key=lambda x: x.get("dph_total", float("inf")))

print(f"{'ID':<10} {'GPU':<18} {'RAM(GB)':<9} {'Price($/h)':<10}\n"+"="*50)
for o in want:
    iid = o.get("id")
    gpu = o.get("gpu_name")
    ram = o.get("gpu_ram")
    price = o.get("dph_total", 0.0)

    # 소수점 3자리까지 포맷
    print(f"{o.get('id',''):<10} {o.get('gpu_name',''):<18} {o.get('gpu_ram',''):<9} {o.get('dph_total',0.0):.3f}")