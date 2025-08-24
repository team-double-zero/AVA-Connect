import control_vm, generate

# vm 부팅 및 연결, 터널링 성공 여부 반환
TUNNEL_OPEN = control_vm.vm_start()

DEBUG = False

# 터널링 성공 시 큐에 대기중인 모든 작업 실행
if TUNNEL_OPEN: 
    try:
        # 큐 길이, 다운로드 수, 전체 시간
        result_img = generate.request_queue('img', debug= DEBUG)
        print(f"[INFO] 이미지 다운로드 시간 {result_img[2]}")

        result_vid = generate.request_queue('vid', debug= DEBUG)
        print(f"[INFO] 비디오 다운로드 시간 {result_vid[2]}")
    except Exception as e:
        print(f"[Error] {e}")

    finally:
        # vm 종료
        print("[INFO] 예약된 작업이 끝났습니다. VM 을 종료합니다.")
        control_vm.vm_stop()