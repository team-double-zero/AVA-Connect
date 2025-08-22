import soruce_codes.get_image as get_image, soruce_codes.vm as vm

# 폴더 img_q 내의 json 파일들 순회 요청
TUNNEL_OPEN = vm.vm_start()
if TUNNEL_OPEN: get_image.request_queue("/img_q")   

print("[INFO] 예약된 작업이 끝났습니다. VM 을 종료합니다.")
vm.vm_stop()