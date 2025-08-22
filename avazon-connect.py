import get_image, vm

TUNNEL_OPEN = vm.vm_start()

if TUNNEL_OPEN: 
    get_image.request_queue("img_q")    # 폴더 img_q 내의 json 파일들 순회 요청

else:
    print("터널링 연결 실패, VM 을 종료합니다.")

vm.vm_stop()