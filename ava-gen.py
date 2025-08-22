from soruce_codes import control_vm, fetch_img, fetch_vid

# 폴더 img_q 내의 json 파일들 순회 요청
TUNNEL_OPEN = control_vm.vm_start()
if TUNNEL_OPEN: 
    fetch_img.request_queue("/q_img")

print("[INFO] 예약된 작업이 끝났습니다. VM 을 종료합니다.")
control_vm.vm_stop()