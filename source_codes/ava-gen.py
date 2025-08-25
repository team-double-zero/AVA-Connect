import argparse

import control_vm, generate

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="자동 종료")
parser.add_argument("--disable-auto-off", action="store_true", help="자동 종료")
args = parser.parse_args()

DEBUG = args.debug
AUTO_OFF = not args.disable_auto_off

# # vm 부팅 및 연결, 터널링 성공 여부 반환
TUNNEL_OPEN = control_vm.vm_start()

# # 터널링 성공 시 큐에 대기중인 모든 작업 실행
if TUNNEL_OPEN: 
    # 큐 길이, 다운로드 수, 전체 시간
    result_img = generate.request_queue('img', debug= DEBUG)
    print(f"[INFO] 이미지 다운로드 시간 {result_img}")

    result_vid = generate.request_queue('vid', debug= DEBUG)
    print(f"[INFO] 비디오 다운로드 시간 {result_vid}")


# # vm 종료
print("[INFO] 예약된 작업이 끝났습니다.")

if AUTO_OFF: print("[OFF] VM 을 종료합니다."); control_vm.vm_stop()
else: print("[INFO] --disable-auto-off 옵션에 의해 자동 종료하지 않습니다.")