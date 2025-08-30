"""Vast.ai helper 모듈: GPU 오퍼 검색 및 인스턴스 관리."""

import os
import math
import time
import warnings
from typing import Any, Tuple, Dict, List, Optional, Union

from vastai_sdk import VastAI

# 상수
POLL_INTERVAL_SECONDS = 5
MB_TO_GB_RATIO = 1024
DEFAULT_OFFER_LIMIT = 200

class VastOffer:
    """Vast.ai 오퍼를 표현하는 클래스.
    
    Attributes:
        id: 오퍼 ID
        ask_contract_id: 인스턴스 생성 시 필요한 ask_id
        gpu_name: GPU 모델명  
        gpu_ram: GPU VRAM 용량 (MB)
        gpu_frac: GPU 할당 비율 (0.0-1.0)
        dph_total: 시간당 총 비용 ($/h)
        dlperf_per_dph: 딥러닝 성능/가격 비율
        reliability: 신뢰도 지수 (0.0-1.0)
        geolocation: 지리적 위치
        cpu_cores: CPU 코어 수
        gpu_ram: 시스템 RAM 용량 (MB)
        disk_space: 디스크 공간 (GB)
        inet_up: 업로드 속도 (Mbps)
        inet_down: 다운로드 속도 (Mbps)
        rentable: 대여 가능 여부
        rented: 현재 대여 중 여부
        verified: 검증된 호스트 여부
        cost_per_vram_gb: VRAM GB당 시간당 비용 (계산된 값)
        vram_gb: VRAM 용량 (GB, 계산된 값)
    """
    
    def __init__(self, offer_data: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None) -> None:
        """Vast.ai 오퍼 데이터로부터 오퍼 객체 생성.
        
        Args:
            offer_data: Vast.ai API에서 받은 오퍼 원본 데이터
            metrics: calculate_metrics에서 계산된 추가 지표 (선택사항)
        """
        # 기본 오퍼 정보
        self.id: Optional[int] = offer_data.get("id")
        self.ask_contract_id: Optional[int] = offer_data.get("ask_contract_id")  # 인스턴스 생성 시 필요
        self.gpu_name: Optional[str] = offer_data.get("gpu_name")
        self.gpu_frac: Optional[float] = offer_data.get("gpu_frac")
        self.dph_total: Optional[float] = offer_data.get("dph_total")
        self.dlperf_per_dph: Optional[float] = offer_data.get("dlperf_per_dphtotal")
        self.reliability: Optional[float] = offer_data.get("reliability")
        self.geolocation: Optional[str] = offer_data.get("geolocation")
        
        # 시스템 스펙
        self.cpu_cores: Optional[int] = offer_data.get("cpu_cores")
        self.gpu_ram: Optional[int] = offer_data.get("gpu_ram")  # MB
        self.disk_space: Optional[float] = offer_data.get("disk_space")  # GB
        
        # 네트워크 정보
        self.inet_up: Optional[float] = offer_data.get("inet_up")  # Mbps
        self.inet_down: Optional[float] = offer_data.get("inet_down")  # Mbps
        
        # 상태 정보
        self.rentable: Optional[bool] = offer_data.get("rentable")
        self.rented: Optional[bool] = offer_data.get("rented")
        self.verified: bool = offer_data.get("verification") == "verified"
        
        # 계산된 지표들 (metrics에서 가져오거나 직접 계산)
        if metrics:
            self.cost_per_vram_gb: float = metrics.get("cost_per_vram_gb", 0.0)
            self.vram_gb: float = metrics.get("vram_gb", 0.0)
        else:
            self.vram_gb: float = (self.gpu_ram or 0) / MB_TO_GB_RATIO
            self.cost_per_vram_gb: float = (
                (self.dph_total or 0) / self.vram_gb if self.vram_gb > 0 else float("inf")
            )
        
        # 원본 오퍼 데이터 보관 (필요시 접근 가능)
        self._raw_data: Dict[str, Any] = offer_data
        
    def __str__(self) -> str:
        """오퍼 정보를 문자열로 반환."""
        return f"VastOffer(id={self.id}, ask_id={self.ask_contract_id}, gpu_name={self.gpu_name}, vram={self.vram_gb:.1f}GB, gpu_frac={self.gpu_frac}, price=${self.dph_total:.3f}/h, location={self.geolocation})"
    
    def __repr__(self) -> str:
        """개발자용 문자열 표현."""
        return self.__str__()
    
    def get_raw_data(self) -> Dict[str, Any]:
        """원본 API 데이터 반환."""
        return self._raw_data

class VastInstance:
    """Vast.ai 인스턴스를 표현하는 클래스.
    
    Attributes:
        id: 인스턴스 ID
        status: 인스턴스 상태 문자열 (소문자)
        gpu_name: GPU 모델명
        gpu_ram: GPU VRAM 용량 (MB)
        gpu_frac: GPU 할당 비율 (0.0-1.0)
        dph_total: 시간당 총 비용 ($/h)
        dlperf_per_dph: 딥러닝 성능 대비 비용 효율성
        cur_state: 현재 실행 상태
        intended_status: 의도된 상태
        ssh_host: SSH 접속 호스트
        ssh_port: SSH 포트 번호  
        public_ipaddr: 공개 IP 주소
        reliability: 신뢰도 지수
        geolocation: 지리적 위치
        ports: 포트 설정 정보
        ssh: SSH 설정 정보
    """
    
    def __init__(self, data: Dict[str, Any]) -> None:
        """Vast.ai API 응답 데이터로부터 인스턴스 객체 생성."""
        # 상태 정보 (우선순위: actual_status > status_msg > status)
        self.status: str = (
            data.get("actual_status") or 
            data.get("status_msg") or 
            data.get("status") or 
            ""
        ).lower()
        
        # 기본 정보
        self.id: Optional[int] = data.get("id")
        self.gpu_name: Optional[str] = data.get("gpu_name")
        self.gpu_ram: Optional[int] = data.get("gpu_ram")  # MB
        self.gpu_frac: Optional[float] = data.get("gpu_frac")
        self.dph_total: Optional[float] = data.get("dph_total")
        self.dlperf_per_dph: Optional[float] = data.get("dlperf_per_dph")
        self.cpu_cores: Optional[int] = data.get("cpu_cores")

        # 추가 시스템/네트워크/상태 정보 및 파생값
        self.cpu_ram: Optional[int] = data.get("cpu_ram")  # MB (호스트 RAM)
        self.disk_space: Optional[float] = data.get("disk_space")  # GB
        self.inet_up: Optional[float] = data.get("inet_up")  # Mbps
        self.inet_down: Optional[float] = data.get("inet_down")  # Mbps
        self.rentable: Optional[bool] = data.get("rentable")
        self.rented: Optional[bool] = data.get("rented")
        self.verified: bool = (data.get("verification") == "verified")

        # 파생값: VRAM(GB) 및 VRAM GB당 비용
        self.vram_gb: float = (self.gpu_ram or 0) / MB_TO_GB_RATIO
        self.cost_per_vram_gb: float = (
            (self.dph_total or 0) / self.vram_gb if self.vram_gb > 0 else float("inf")
        )
        
        # 실행 상태
        self.cur_state: Optional[str] = data.get("cur_state")
        self.intended_status: Optional[str] = data.get("intended_status")
        
        # 네트워크 정보
        self.public_ipaddr: Optional[str] = data.get("public_ipaddr")
        
        # 메타데이터
        self.reliability: Optional[float] = data.get("reliability2")
        self.geolocation: Optional[str] = data.get("geolocation")
        
        # SSH 정보 (제거됨)

    # extract_ssh_info 메서드 제거됨
    
    def __str__(self) -> str:
        """인스턴스 정보를 문자열로 반환."""
        return (
            f"VastInstance(id={self.id}, gpu_name={self.gpu_name}, "
            f"gpu_ram={self.gpu_ram}, gpu_frac={self.gpu_frac}, dph_total={self.dph_total}, "
            f"dlperf_per_dph={self.dlperf_per_dph}, reliability={self.reliability}, geolocation={self.geolocation}, "
            f"cpu_cores={self.cpu_cores}, disk_space={getattr(self, 'disk_space', None)}, "
            f"inet_up={getattr(self, 'inet_up', None)}, inet_down={getattr(self, 'inet_down', None)}, "
            f"rentable={getattr(self, 'rentable', None)}, rented={getattr(self, 'rented', None)}, "
            f"verified={getattr(self, 'verified', None)}, cost_per_vram_gb={getattr(self, 'cost_per_vram_gb', None)}, "
            f"vram_gb={getattr(self, 'vram_gb', None)}, cur_state={self.cur_state})"
        )
    
    def __repr__(self) -> str:
        """개발자용 문자열 표현."""
        return self.__str__()

class VastHelper:
    """Vast.ai 헬퍼: 내부적으로 VastAI 클라이언트를 관리하고, 오퍼 검색/랭킹 제공."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """VastHelper 초기화.
        
        Args:
            api_key: Vast.ai API 키. None이면 VAST_API_KEY 환경변수 사용
        """
        # 환경 변수에서 키를 읽되, 이 단계에서는 .env 로딩을 수행하지 않음
        self.api_key: Optional[str] = api_key or os.getenv("VAST_API_KEY")
        self.client: Optional[VastAI] = None
        
        if not self.api_key:
            warnings.warn("VAST_API_KEY가 설정되지 않았습니다. VastHelper는 API 키가 필요합니다.", UserWarning, stacklevel=2)
            return
            
        try:
            self.client = VastAI(api_key=self.api_key)
        except Exception as exc:
            warnings.warn(
                f"VastAI 클라이언트 초기화 실패: {exc}",
                RuntimeWarning,
                stacklevel=2
            )
            self.client = None

    def check_client(self, print_output: bool = False) -> bool:
        """클라이언트가 정상적으로 초기화되었는지 확인
        
        Args:
            print_output (bool): 에러 발생 시 출력할지 여부
            
        Returns:
            bool: 클라이언트가 사용 가능하면 True, 아니면 False
        """
        if self.client is None:
            if print_output:
                if not self.api_key:
                    print("❌ VastHelper 초기화 실패: VAST_API_KEY가 설정되지 않았습니다.")
                else:
                    print("❌ VastHelper 초기화 실패: VastAI 클라이언트 생성에 실패했습니다.")
            return False
        return True

    @staticmethod
    def calculate_metrics(offer: Dict[str, Any]) -> Dict[str, Any]:
        """오퍼의 가성비 지표들을 계산.
        
        Args:
            offer: Vast.ai 오퍼 데이터
            
        Returns:
            가성비 지표가 포함된 딕셔너리
        """
        dph = offer.get("dph_total", float("inf"))
        vram_gb = offer.get("gpu_ram", 0) / MB_TO_GB_RATIO  # MB to GB
        dlperf_per_dph = offer.get("dlperf_per_dphtotal", 0.0)
        reliability = offer.get("reliability", 0.0)
        gpu_frac = offer.get("gpu_frac", 1.0)

        cost_per_vram_gb = dph / vram_gb if vram_gb > 0 else float("inf")

        return {
            'offer': offer,
            'dph_total': dph,
            'vram_gb': vram_gb,
            'dlperf_per_dph': dlperf_per_dph,
            'reliability': reliability,
            'cost_per_vram_gb': cost_per_vram_gb,
            'gpu_frac': gpu_frac
        }

    @staticmethod
    def comprehensive_score(
        metrics: Dict[str, Any],
        weight_dlperf: float = 0.6,
        weight_reliability: float = 0.2,
        weight_gpu_frac: float = 0.2,
        reliability_scale: float = 20.0,
        gpu_frac_scale: float = 20.0,
    ) -> float:
        """종합 점수 계산 (딥러닝 성능, 신뢰도, GPU 전용도 고려).
        
        Args:
            metrics: calculate_metrics로 계산된 지표 딕셔너리
            weight_dlperf: dlperf_per_dphtotal 가중치
            weight_reliability: reliability 가중치  
            weight_gpu_frac: gpu_frac 가중치
            reliability_scale: reliability 스케일링 계수
            gpu_frac_scale: gpu_frac 스케일링 계수
            
        Returns:
            종합 점수 (높을수록 좋음)
        """
        dlperf_term = metrics['dlperf_per_dph'] * weight_dlperf
        reliability_term = metrics['reliability'] * reliability_scale * weight_reliability
        gpu_frac_term = metrics['gpu_frac'] * gpu_frac_scale * weight_gpu_frac
        return dlperf_term + reliability_term + gpu_frac_term

    def find_best_offer(
        self,
        *,
        print_output: bool = False,
        gpu_model: str = "A100",
        min_vram_mb: int = 0,
        min_gpu_frac: float = 0.0,
        require_single_gpu: bool = True,
        weight_dlperf: float = 0.6,
        weight_reliability: float = 0.2,
        weight_gpu_frac: float = 0.2,
        reliability_scale: float = 20.0,
        gpu_frac_scale: float = 20.0,
    ) -> Optional[VastOffer]:
        """지정된 GPU 모델 중에서 최고 가성비 오퍼 찾기
        
        Args:
            print_output (bool): True면 분석 과정 출력, False면 결과만 반환
            gpu_model (str): 검색할 GPU 모델명 (예: "A100", "H100", "RTX4090")
            min_vram_mb (int): 최소 VRAM 용량 (MB 단위, 40960 = 40GB)
            min_gpu_frac (float): 이 값 이상(gpu_frac >= min_gpu_frac)인 오퍼만 선택
            require_single_gpu (bool): True이면 num_gpus=1 오퍼만 선택
            weight_dlperf (float): dlperf_per_dphtotal 가중치
            weight_reliability (float): reliability 가중치
            weight_gpu_frac (float): gpu_frac 가중치
            reliability_scale (float): reliability 스케일링 계수
            gpu_frac_scale (float): gpu_frac 스케일링 계수
        """

        if not self.check_client(print_output=print_output):
            return None

        try:
            base_query = f"gpu_name~{gpu_model} rentable=true rented=false verified=true"
            if require_single_gpu:
                base_query += " num_gpus=1"
            offers = self.client.search_offers(
                query=base_query,
                limit=DEFAULT_OFFER_LIMIT,
            )
        except Exception as exc:
            if print_output:
                print(f"❌ 오퍼 조회 중 오류가 발생했습니다: {exc}")
            return None

        # 필터링 조건에 맞는 오퍼만 선택
        filtered_offers = [
            offer for offer in offers
            if (
                "gpu_name" in offer
                and gpu_model.lower() in offer["gpu_name"].lower()
                and offer.get("gpu_ram", 0) >= min_vram_mb
                and offer.get("gpu_frac", 1.0) >= min_gpu_frac
                and (offer.get("num_gpus", 1) == 1 if require_single_gpu else True)
            )
        ]

        if not filtered_offers:
            if print_output:
                print(f"❌ 조건에 맞는 {gpu_model} GPU 오퍼를 찾을 수 없습니다.")
            return None

        metrics = [self.calculate_metrics(offer) for offer in filtered_offers]

        # 종합 점수 기준으로 최적 GPU 선택
        best_overall = sorted(
            metrics,
            key=lambda m: self.comprehensive_score(
                metrics=m,
                weight_dlperf=weight_dlperf,
                weight_reliability=weight_reliability,
                weight_gpu_frac=weight_gpu_frac,
                reliability_scale=reliability_scale,
                gpu_frac_scale=gpu_frac_scale,
            ),
            reverse=True,
        )

        if best_overall:
            top_metrics = best_overall[0]
            best_offer = VastOffer(top_metrics['offer'], top_metrics)
            
            if print_output:
                print(f"✅ 추천 GPU: {best_offer.gpu_name} (ID: {best_offer.id}) - ${best_offer.dph_total:.3f}/h")

            return best_offer
        else:
            if print_output:
                print("❌ 조건에 맞는 GPU를 찾을 수 없습니다.")
            return None

    def get_instances(self) -> Optional[List[VastInstance]]:
        """사용자의 모든 인스턴스 목록 조회.
        
        Returns:
            VastInstance 객체들의 리스트. 실패 시 None
        """
        if not self.check_client():
            return None
            
        try:
            instances_data = self.client.show_instances()
            return [VastInstance(instance_data) for instance_data in instances_data]
        except Exception as exc:
            warnings.warn(
                f"인스턴스 목록 조회 실패: {exc}",
                RuntimeWarning,
                stacklevel=2
            )
            return None

    def launch_instance_by_offer(
        self,
        offer: VastOffer,
        *,
        image: Optional[str] = "auto",
        disk_gb: int = 100,
        ssh: bool = True,
    ) -> VastInstance:
        """선택한 오퍼로 인스턴스를 생성하여 반환합니다.

        기본 파라미터는 다음과 같습니다:
        - image: "auto"면 오퍼의 cuda_max_good에 맞춘 권장 이미지 선택
        - disk_gb: 디스크 크기 GB (기본값 100)
        - ssh: SSH 활성화 (기본값 True)
        """
        if not self.check_client():
            raise RuntimeError("VastAI 클라이언트가 초기화되지 않았습니다.")

        # 이미지 자동 선택 (vastai/base-image:cuda-<ver>-auto)
        if image == "auto":
            image = self._select_auto_cuda_image(offer)

        # SDK는 create_instance(id=...) 형태를 요구함
        contract_id = offer.ask_contract_id or offer.id
        if not contract_id:
            raise ValueError("offer.ask_contract_id 또는 offer.id 가 없습니다. 인스턴스를 생성할 수 없습니다.")

        # SDK 시그니처가 ask_id 또는 ask_contract_id 를 받을 수 있어 둘 다 시도
        instance_data: Optional[Dict[str, Any]] = None
        try:
            instance_data = self.client.create_instance(  # type: ignore[assignment]
                id=int(contract_id),
                image=image,
                disk=float(disk_gb),
                ssh=ssh,
            )
        except Exception as exc:
            raise RuntimeError(f"create_instance 실패(id={contract_id}): {exc}")

        if not instance_data:
            raise RuntimeError("create_instance 결과가 비어 있습니다.")

        # 가능하면 최신 상세 정보를 가져와 래핑
        try:
            iid = int(instance_data.get("id")) if isinstance(instance_data.get("id"), (int, str)) else None
        except Exception:
            iid = None

        if iid is not None:
            try:
                latest = self.client.show_instance(id=iid)
                if isinstance(latest, dict) and latest:
                    return VastInstance(latest)
            except Exception as exc:
                warnings.warn(f"show_instance(id={iid}) 호출 실패: {exc}", RuntimeWarning, stacklevel=2)

        return VastInstance(instance_data)

    @staticmethod
    def _select_auto_cuda_image(offer: VastOffer) -> str:
        """오퍼의 cuda_max_good에 맞춰 권장 CUDA 이미지 태그 선택.

        정책: cuda_max_good을 소수 첫째자리로 내림해 사용.
        최종 태그: vastai/base-image:cuda-<ver>-auto (예: 12.8 → 12.8)
        """
        raw = offer.get_raw_data() if hasattr(offer, "get_raw_data") else {}
        cmg = raw.get("cuda_max_good")
        try:
            cmg_val = float(cmg) if cmg is not None else None
        except Exception:
            cmg_val = None

        if cmg_val is None:
            # 정보가 없으면 보편적인 12.6 사용
            tag_ver = "12.6"
        else:
            floored = math.floor(cmg_val * 10.0) / 10.0
            tag_ver = f"{floored:.1f}"
        return f"vastai/base-image:cuda-{tag_ver}-auto"

    # ---- Basic instance controls ----
    def stop_instance(self, instance: VastInstance) -> bool:
        """인스턴스 정지."""
        if not self.check_client():
            return False
        if instance.id is None:
            raise ValueError("instance.id 가 없습니다.")
        try:
            self.client.stop_instance(id=int(instance.id))
            return True
        except Exception as exc:
            warnings.warn(f"stop_instance 실패(id={instance.id}): {exc}", RuntimeWarning, stacklevel=2)
            return False

    def destroy_instance(self, instance: VastInstance) -> bool:
        """인스턴스 삭제(파괴)."""
        if not self.check_client():
            return False
        if instance.id is None:
            raise ValueError("instance.id 가 없습니다.")
        try:
            self.client.destroy_instance(id=int(instance.id))
            return True
        except Exception as exc:
            warnings.warn(f"destroy_instance 실패(id={instance.id}): {exc}", RuntimeWarning, stacklevel=2)
            return False

    def reboot_instance(self, instance: VastInstance) -> bool:
        """인스턴스 재부팅."""
        if not self.check_client():
            return False
        if instance.id is None:
            raise ValueError("instance.id 가 없습니다.")
        try:
            self.client.reboot_instance(id=int(instance.id))
            return True
        except Exception as exc:
            warnings.warn(f"reboot_instance 실패(id={instance.id}): {exc}", RuntimeWarning, stacklevel=2)
            return False

    def start_instance(self, instance: VastInstance) -> bool:
        """인스턴스 시작."""
        if not self.check_client():
            return False
        if instance.id is None:
            raise ValueError("instance.id 가 없습니다.")
        try:
            self.client.start_instance(id=int(instance.id))
            return True
        except Exception as exc:
            warnings.warn(f"start_instance 실패(id={instance.id}): {exc}", RuntimeWarning, stacklevel=2)
            return False
        
    def wait_boot_instance(self, instance: VastInstance, max_time_out: int = 24) -> VastInstance:
        self.start_instance(instance)
        print(f"[INFO] Trying to boot {instance.id}")
        for _ in range(max_time_out):
            i = self.client.show_instance(id= instance.id)
            status = i.get("actual_status")
            if status == "running": 
                print(f"[INFO] Now on {status}")
                return True
            time.sleep(5)
        self.destroy_instance(instance)
        return False

    def get_ssh_info(self, instance: VastInstance):
        inst = self.client.show_instance(id= instance.id)
        
        # SSH 연결정보 파싱
        ssh = inst.get("ssh", {}) or {}
        host = inst.get("ssh_host")
        port = ssh.get("port") or inst.get("ssh_port")

        # 포트 매핑 폴백
        if not port:
            ports = inst.get("ports", {}) or {}
            port_map = inst.get("port_map", {}) or {}
            port = ports.get("22/tcp", {}).get("HostPort") or port_map.get("22/tcp")

        return host, port


    def run_best_instance(self):
        best_instance = None
        owned_instances = self.get_instances()
        host, port = None, None
        
        while owned_instances:
            owned_instance = owned_instances.pop(0)
            if self.wait_boot_instance(owned_instance): 
                best_instance = owned_instance

        if not best_instance:
            best_offer = None
            while not best_offer:
                best_offer = self.find_best_offer(
                    print_output=True,
                    gpu_model="A100",
                    min_vram_mb=40960,  # 40GB+
                    min_gpu_frac=0.5
                )
                time.sleep(5)
            new_instance = self.launch_instance_by_offer(offer= best_offer)
            if self.wait_boot_instance(new_instance): 
                best_instance = new_instance
        
        if best_instance: host, port = self.get_ssh_info(best_instance)
        
        return host, port, best_instance
