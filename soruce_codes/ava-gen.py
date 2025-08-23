import control_vm, fetch_img, fetch_vid

# vm 부팅 및 연결, 터널링 성공 여부 반환
TUNNEL_OPEN = control_vm.vm_start()

# 터널링 성공 시 큐에 대기중인 모든 작업 실행
if TUNNEL_OPEN: 
    fetch_img.request_queue("../q_img")
    fetch_vid.request_queue("../q_vid")

# vm 종료
control_vm.vm_stop()
print("[INFO] 예약된 작업이 끝났습니다. VM 을 종료합니다.")