"""Vast.ai helper 모듈: GPU 오퍼 검색 및 인스턴스 관리."""

import os
import math
import time
import warnings
import subprocess
import paramiko
import socket
import re
import requests
from typing import Any, Tuple, Dict, List, Optional, Union

from vastai_sdk import VastAI
from retry_decorator import with_api_retry, with_ssh_retry, retry_on_failure

# 상수
POLL_INTERVAL_SECONDS = 5
MB_TO_GB_RATIO = 1024
DEFAULT_OFFER_LIMIT = 200

# Vast.ai base-image에서 안정적으로 제공되는 CUDA auto 태그(검증/안전한 범위)
# 특정 호스트의 cuda_max_good 값이 미묘한 소수(예: 12.2)로 떨어질 수 있어
# 존재하지 않는 태그를 요청하여 "manifest unknown" 오류가 나는 것을 방지한다.
SUPPORTED_CUDA_AUTO_VERSIONS = [  # 네트워크 실패 시 최소한의 폴백용
    "12.8", "12.7", "12.6", "12.5", "12.4", "12.3", "11.8"
]
DEFAULT_CUDA_AUTO_VERSION = "12.6"  # 폴백 기본값

# Docker Hub 태그 캐시 (1시간 TTL)
_docker_tags_cache = {
    "ts": 0.0,
    "versions": []  # ["12.8", "12.6", ...]
}

# ===== 부팅/SSH 타이밍 상수 =====
# 환경변수가 아닌 코드 상단 상수로 고정 관리
BOOT_TIMEOUT_TOTAL = 300            # 전체 부팅 타임아웃(초)
BOOT_SSH_EXTRA_WAIT = 5            # running 이후 SSH 서비스 여유 대기(초)
BOOT_SSH_VERIFY_ATTEMPTS = 30       # SSH 검증 루프 시도 횟수(초 단위)

def _parse_cuda_auto_versions_from_dockerhub_response(results: list) -> list:
    pattern = re.compile(r"^cuda-(\d+\.\d+)-auto$")
    versions = []
    for entry in results:
        name = entry.get("name", "")
        m = pattern.match(name)
        if m:
            versions.append(m.group(1))
    # 중복 제거 및 내림차순 정렬
    unique = sorted({v for v in versions}, key=lambda x: float(x), reverse=True)
    return unique

def fetch_vast_base_image_cuda_auto_tags(force_refresh: bool = False, timeout: int = 8) -> list:
    """Docker Hub에서 vastai/base-image의 cuda-*-auto 태그 목록을 조회.

    - 환경변수 `VAST_CUDA_AUTO_TAGS`로 수동 지정 가능(쉼표 구분)
    - 네트워크 실패 시 캐시 또는 폴백 목록 사용
    """
    try:
        override = os.getenv("VAST_CUDA_AUTO_TAGS")
        if override:
            manual = [v.strip() for v in override.split(",") if v.strip()]
            # 숫자 버전만 유지
            manual = [v for v in manual if re.match(r"^\d+\.\d+$", v)]
            if manual:
                return sorted({*manual}, key=lambda x: float(x), reverse=True)

        now = time.time()
        if not force_refresh and _docker_tags_cache["versions"] and (now - _docker_tags_cache["ts"]) < 3600:
            return list(_docker_tags_cache["versions"])

        url = "https://hub.docker.com/v2/repositories/vastai/base-image/tags?page_size=100"
        versions_accum = []
        # 페이지네이션 따라가며 수집
        while url:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            versions_accum.extend(_parse_cuda_auto_versions_from_dockerhub_response(results))
            url = data.get("next")

        if versions_accum:
            # 최신 전체 목록 정리 및 캐시
            versions = sorted({*versions_accum}, key=lambda x: float(x), reverse=True)
            _docker_tags_cache["versions"] = versions
            _docker_tags_cache["ts"] = time.time()
            return versions

    except Exception:
        # 네트워크/파싱 실패 시 캐시 또는 폴백
        if _docker_tags_cache["versions"]:
            return list(_docker_tags_cache["versions"])

    # 최종 폴백: 하드코딩 최소 목록
    return list(SUPPORTED_CUDA_AUTO_VERSIONS)

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

    @with_api_retry
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

    @with_api_retry
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
        disk_gb: int = 130,
        ssh: bool = True,
        ensure_ssh_keys: bool = True,
    ) -> VastInstance:
        """선택한 오퍼로 인스턴스를 생성하여 반환합니다.

        기본 파라미터는 다음과 같습니다:
        - image: "auto"면 오퍼의 cuda_max_good에 맞춘 권장 이미지 선택
        - disk_gb: 디스크 크기 GB (기본값 130)
        - ssh: SSH 활성화 (기본값 True)
        - ensure_ssh_keys: 인스턴스 부팅 후 팀 SSH 키 등록 여부 (기본값 True)
        """
        if not self.check_client():
            raise RuntimeError("VastAI 클라이언트가 초기화되지 않았습니다.")

        # SSH 키 설정 검증
        ssh_key_path = os.getenv("SSH_KEY_PATH")
        if ssh and (not ssh_key_path or not os.path.exists(ssh_key_path)):
            raise RuntimeError(f"SSH 키 파일을 찾을 수 없습니다: {ssh_key_path}")

        # 이미지 자동 선택 (vastai/base-image:cuda-<ver>-auto)
        if image == "auto":
            image = self._select_auto_cuda_image(offer)

        print(f"[LAUNCH] 인스턴스 생성 중... (오퍼ID: {offer.id}, 이미지: {image})")

        # SDK는 create_instance(id=...) 형태를 요구함
        contract_id = offer.ask_contract_id or offer.id
        if not contract_id:
            raise ValueError("offer.ask_contract_id 또는 offer.id 가 없습니다. 인스턴스를 생성할 수 없습니다.")

        # SDK 시그니처가 ask_id 또는 ask_contract_id 를 받을 수 있어 둘 다 시도
        instance_data: Optional[Dict[str, Any]] = None
        try:
            # SSH 키와 함께 인스턴스 생성
            create_params = {
                "id": int(contract_id),
                "image": image,
                "disk": float(disk_gb),
                "ssh": ssh,
            }
            
            # SSH 키 명시적 추가 (Vast.ai 계정의 모든 SSH 키가 자동 포함됨)
            print(f"[LAUNCH] 생성 파라미터: {create_params}")
            
            instance_data = self.client.create_instance(**create_params)  # type: ignore[assignment]
            print(f"[LAUNCH] 인스턴스 생성 요청 완료: {instance_data}")
            
        except Exception as exc:
            raise RuntimeError(f"create_instance 실패(id={contract_id}): {exc}")

        if not instance_data:
            raise RuntimeError("create_instance 결과가 비어 있습니다.")

        # 인스턴스 생성 직후 잠시 대기 (API 상태 동기화)
        print(f"[LAUNCH] 인스턴스 정보 동기화 대기 중... (5초)")
        time.sleep(5)  # API 상태 동기화를 위한 최소 대기

        # 인스턴스 ID 추출 (여러 응답 형태 지원)
        instance_id = None
        try:
            # 표준 응답 형태: {"id": 123}
            if "id" in instance_data:
                instance_id = int(instance_data["id"])
            # Vast.ai 생성 응답 형태: {"success": True, "new_contract": 123}
            elif "new_contract" in instance_data and instance_data.get("success"):
                instance_id = int(instance_data["new_contract"])
                print(f"[LAUNCH] new_contract ID 사용: {instance_id}")
            else:
                print(f"[LAUNCH] 알 수 없는 응답 형태: {instance_data}")
        except (ValueError, TypeError) as e:
            print(f"[LAUNCH] ID 추출 실패: {e}")

        # ID를 찾은 경우 최신 인스턴스 정보 조회
        instance = None
        if instance_id is not None:
            try:
                latest = self.client.show_instance(id=instance_id)
                if isinstance(latest, dict) and latest:
                    instance = VastInstance(latest)
                    print(f"[LAUNCH] 인스턴스 생성 완료: ID={instance.id}")
                else:
                    print(f"[LAUNCH] show_instance 응답이 비어있음: {latest}")
            except Exception as exc:
                print(f"[LAUNCH] show_instance 실패: {exc}")
                # 실패해도 기본 데이터로 인스턴스 생성 시도

        # 최후의 수단: 원본 데이터에 ID 수동 추가하여 인스턴스 생성
        if instance is None:
            if instance_id is not None and "id" not in instance_data:
                instance_data = dict(instance_data)  # 복사본 생성
                instance_data["id"] = instance_id
                print(f"[LAUNCH] 수동으로 ID 추가하여 인스턴스 생성: {instance_id}")
            
            instance = VastInstance(instance_data)

        if ensure_ssh_keys and instance is not None:
            self.add_ssh_keys_to_instance(instance)

        return instance

    @staticmethod
    def _select_auto_cuda_image(offer: VastOffer) -> str:
        """오퍼의 cuda_max_good에 맞춰 권장 CUDA 이미지 태그 선택.

        - 기본 정책: cuda_max_good을 소수 첫째자리로 내림해 후보 버전을 계산.
        - 안정성 보강: 실제 레지스트리에 존재하는 안전한 버전 집합으로 매핑.
          존재하지 않는 버전(예: 12.2)이 나오면, 그 이하에서 가장 가까운
          지원 버전 또는 기본값(12.6)으로 보정한다.

        반환 예: "vastai/base-image:cuda-12.6-auto"
        """
        raw = offer.get_raw_data() if hasattr(offer, "get_raw_data") else {}
        cmg = raw.get("cuda_max_good")
        try:
            cmg_val = float(cmg) if cmg is not None else None
        except Exception:
            cmg_val = None

        # 1) 후보 버전 계산
        if cmg_val is None:
            candidate = DEFAULT_CUDA_AUTO_VERSION
        else:
            floored = math.floor(cmg_val * 10.0) / 10.0
            candidate = f"{floored:.1f}"

        # 2) 실제 Docker Hub에서 cuda-*-auto 지원 버전 조회
        available = fetch_vast_base_image_cuda_auto_tags()
        if not available:
            available = list(SUPPORTED_CUDA_AUTO_VERSIONS)

        try:
            candidate_f = float(candidate)
        except Exception:
            candidate_f = float(DEFAULT_CUDA_AUTO_VERSION)

        # 3) 하향 근사: 후보 이하에서 가장 가까운 값
        lower_or_equal = [v for v in available if float(v) <= candidate_f]
        if lower_or_equal:
            tag_ver = max(lower_or_equal, key=lambda x: float(x))
        else:
            # 4) 상향 근사: 후보보다 큰 값 중 가장 작은 값
            higher = [v for v in available if float(v) > candidate_f]
            if higher:
                tag_ver = min(higher, key=lambda x: float(x))
            else:
                # 5) 최종 폴백
                tag_ver = DEFAULT_CUDA_AUTO_VERSION

        # 3) 최종 태그 구성
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
        
    def wait_boot_instance(self, instance: VastInstance, max_time_out: int = 50, ensure_ssh_keys: bool = True) -> bool:
        """인스턴스 부팅 및 SSH 키 설정 완료까지 대기
        
        Args:
            instance: 부팅할 인스턴스
            max_time_out: 최대 대기 시간(초)
            ensure_ssh_keys: 팀 SSH 키 등록 여부 (기본값 True)
        """
        # 상단 상수 사용
        max_time_out = BOOT_TIMEOUT_TOTAL

        self.start_instance(instance)
        print(f"[INFO] Trying to boot {instance.id} (timeout={max_time_out}s)")
        
        # 1단계: 인스턴스 running 상태까지 대기
        boot_timeout = max_time_out // 2  # 부팅에 절반 시간 할당
        for attempt in range(boot_timeout):
            i = self.client.show_instance(id=instance.id)
            status = i.get("actual_status")
            print(f"[BOOT-CHECK] Instance status: '{status}' (attempt {attempt + 1}/{boot_timeout})")
            
            if status == "running":
                print(f"[INFO] Instance running (attempt {attempt + 1}/{boot_timeout})")
                # running이 되어도 SSH 서비스가 완전히 시작될 때까지 추가 대기
                print(f"[INFO] SSH 서비스 시작 대기 중... ({BOOT_SSH_EXTRA_WAIT}초)")
                time.sleep(BOOT_SSH_EXTRA_WAIT)
                break
            elif status == "failed":
                print(f"[ERROR] Instance failed to boot")
                self.destroy_instance(instance)
                return False
            time.sleep(5)
        else:
            print(f"[ERROR] Boot timeout after {boot_timeout * 5} seconds")
            self.destroy_instance(instance)
            return False
        
        # 2단계: SSH 키 설정 완료 확인 (더 빠른 검증)
        # SSH 검증 시도 횟수(초 단위 루프)
        ssh_timeout = BOOT_SSH_VERIFY_ATTEMPTS
        if self._verify_ssh_setup(instance, timeout=ssh_timeout):
            print(f"[INFO] SSH setup verified successfully")
            
            # 3단계: SSH 검증 완료 후 팀 SSH 키들 자동 추가
            if ensure_ssh_keys:
                print(f"[INFO] 팀 SSH 키 등록 시작...")
                try:
                    if self.add_ssh_keys_to_instance(instance):
                        print(f"[INFO] ✅ 팀 SSH 키 등록 완료")
                    else:
                        print(f"[INFO] ⚠️  팀 SSH 키 등록 실패 (기본 키로만 접속 가능)")
                except Exception as e:
                    print(f"[INFO] ❌ 팀 SSH 키 등록 중 오류: {e}")
            else:
                print(f"[INFO] 팀 SSH 키 등록 건너뛰기 (ensure_ssh_keys=False)")
            
            return True
        else:
            print(f"[ERROR] SSH setup verification failed")
            # SSH 설정 실패 시 수동 복구 시도 (더 빠르게)
            if self._recover_ssh_setup(instance):
                # 복구 성공 시에도 팀 SSH 키 추가 시도
                if ensure_ssh_keys:
                    try:
                        print(f"[RECOVER] 복구 후 팀 SSH 키 등록 시작...")
                        if self.add_ssh_keys_to_instance(instance):
                            print(f"[RECOVER] ✅ 팀 SSH 키 등록 완료")
                        else:
                            print(f"[RECOVER] ⚠️  팀 SSH 키 등록 실패")
                    except Exception as e:
                        print(f"[RECOVER] 팀 SSH 키 등록 중 오류: {e}")
                else:
                    print(f"[RECOVER] 팀 SSH 키 등록 건너뛰기 (ensure_ssh_keys=False)")
                return True
            
            self.destroy_instance(instance)
            return False

    def get_ssh_info(self, instance: VastInstance):
        inst = self.client.show_instance(id=instance.id)
        host = inst.get("ssh_host")
        port = inst.get("ssh_port")
        
        # 간단한 포트 보정 - API 포트가 틀릴 수 있으므로 ±2 범위 시도
        if port:
            # 원본 포트와 ±1, ±2 포트를 시도
            port_candidates = [port, port + 1, port - 1, port + 2, port - 2]
            for test_port in port_candidates:
                # 간단한 소켓 연결 테스트
                import socket
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    result = sock.connect_ex((host, test_port))
                    sock.close()
                    if result == 0:  # 연결 성공
                        return host, test_port
                except:
                    continue
        
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

    @retry_on_failure(max_attempts=2, delay=3.0, log_attempts=True, return_on_final_failure=False)
    def _verify_ssh_setup(self, instance: VastInstance, timeout: int = 25) -> bool:
        """SSH 키 설정이 완료되었는지 검증"""
        print(f"[SSH-VERIFY] SSH 설정 검증 시작 (타임아웃: {timeout * 5}초)")
        
        # SSH 연결 정보 가져오기
        try:
            host, port = self.get_ssh_info(instance)
            if not host or not port:
                print("[SSH-VERIFY] SSH 연결 정보를 가져올 수 없음")
                return False
        except Exception as e:
            print(f"[SSH-VERIFY] SSH 정보 조회 실패: {e}")
            return False

        ssh_key_path = os.getenv("SSH_KEY_PATH")
        if not ssh_key_path or not os.path.exists(ssh_key_path):
            print("[SSH-VERIFY] SSH 키 파일을 찾을 수 없음")
            return False

        def _banner_ready(h: str, p: int, wait: float = 3.0) -> bool:
            try:
                with socket.create_connection((h, p), timeout=wait) as s:
                    s.settimeout(wait)
                    data = s.recv(64)
                    return data.startswith(b"SSH-")
            except Exception:
                return False

        # 간단한 SSH 연결 시도 (배너 준비 선확인 + 여유 타임아웃)
        for attempt in range(timeout):
            try:
                print(f"[SSH-VERIFY] 연결 시도 {attempt + 1}/{timeout}: {host}:{port}")
                if not _banner_ready(host, port, wait=4.0):
                    print("[SSH-VERIFY] 배너 미준비 - 대기 후 재시도")
                    time.sleep(2)
                    continue
                
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(
                    hostname=host,
                    port=port,
                    username="root",
                    key_filename=ssh_key_path,
                    timeout=12,
                    banner_timeout=20,
                    auth_timeout=20,
                    look_for_keys=False,
                    allow_agent=False
                )
                ssh.close()
                
                print(f"[SSH-VERIFY] ✅ SSH 연결 성공!")
                return True
                    
            except Exception as e:
                print(f"[SSH-VERIFY] 연결 실패: {e}")
                time.sleep(2)
        
        print(f"[SSH-VERIFY] ❌ SSH 연결 실패")
        return False

    def _recover_ssh_setup(self, instance: VastInstance) -> bool:
        """SSH 설정 실패 시 복구 시도"""
        print("[SSH-RECOVER] SSH 설정 복구 시도 중...")
        
        try:
            # 1. 인스턴스 재부팅 시도
            print("[SSH-RECOVER] 인스턴스 재부팅 중...")
            if self.reboot_instance(instance):
                time.sleep(30)  # 재부팅 대기
                
                # 재부팅 후 검증
                if self._verify_ssh_setup(instance, timeout=15):
                    print("[SSH-RECOVER] ✅ 재부팅 후 SSH 복구 성공")
                    return True
            
            # 2. SSH 키 수동 추가 시도 (Vast.ai API 통해)
            print("[SSH-RECOVER] SSH 키 수동 추가 시도...")
            if self._add_ssh_keys_to_instance(instance):
                time.sleep(10)  # 키 추가 대기
                
                if self._verify_ssh_setup(instance, timeout=10):
                    print("[SSH-RECOVER] ✅ SSH 키 추가 후 복구 성공")
                    return True
                    
        except Exception as e:
            print(f"[SSH-RECOVER] 복구 과정 중 오류: {e}")
        
        print("[SSH-RECOVER] ❌ SSH 복구 실패")
        return False

    def get_account_ssh_keys(self) -> List[str]:
        """계정에 등록된 모든 SSH 키 목록 조회"""
        if not self.check_client():
            return []
            
        ssh_keys = []
        
        # 방법 1: SDK를 통한 조회 시도
        try:
            # show_ssh_keys 메서드 사용
            keys_response = self.client.show_ssh_keys()
            if keys_response and isinstance(keys_response, list):
                for key_info in keys_response:
                    if isinstance(key_info, dict):
                        # SSH 키 내용 추출
                        if "public_key" in key_info:
                            ssh_keys.append(key_info["public_key"])
                        elif "ssh_key" in key_info:
                            ssh_keys.append(key_info["ssh_key"])
                        elif "key" in key_info:
                            ssh_keys.append(key_info["key"])
                    elif isinstance(key_info, str):
                        ssh_keys.append(key_info)
                        
            print(f"[SSH-KEYS] SDK를 통해 {len(ssh_keys)}개 SSH 키 조회")
                
        except Exception as e:
            print(f"[SSH-KEYS] SDK 조회 실패: {e}")
        
        print(f"[SSH-KEYS] 총 {len(ssh_keys)}개 SSH 키 수집 완료")
        for i, key in enumerate(ssh_keys):
            key_preview = key[:50] + "..." if len(key) > 50 else key
            print(f"[SSH-KEYS] Key {i+1}: {key_preview}")
        
        return ssh_keys

    def _get_existing_ssh_keys(self, instance: VastInstance) -> List[str]:
        """인스턴스에 현재 등록된 SSH 키들 조회"""
        try:
            host, port = self.get_ssh_info(instance)
            ssh_key_path = os.getenv("SSH_KEY_PATH")
            
            if not host or not port or not ssh_key_path or not os.path.exists(ssh_key_path):
                print("[SSH-CHECK] SSH 연결 정보 또는 키 파일을 찾을 수 없음")
                return []
            
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                hostname=host,
                port=port,
                username="root",
                key_filename=ssh_key_path,
                timeout=10,
                look_for_keys=False
            )
            
            # authorized_keys 파일 내용 조회
            stdin, stdout, stderr = ssh.exec_command("cat ~/.ssh/authorized_keys 2>/dev/null || echo ''", timeout=10)
            authorized_keys_content = stdout.read().decode().strip()
            ssh.close()
            
            if authorized_keys_content:
                # 각 라인이 SSH 키
                existing_keys = [line.strip() for line in authorized_keys_content.split('\n') if line.strip()]
                print(f"[SSH-CHECK] authorized_keys에서 {len(existing_keys)}개 키 발견")
                return existing_keys
            else:
                print("[SSH-CHECK] authorized_keys 파일이 비어있거나 없음")
                return []
                
        except Exception as e:
            print(f"[SSH-CHECK] 기존 키 조회 실패: {e}")
            return []

    def _clean_instance_ssh_keys(self, instance: VastInstance) -> bool:
        """인스턴스에서 모든 SSH 키 제거 (Vast.ai 계정 키 + 로컬 키)"""
        print(f"[SSH-CLEAN] 인스턴스 {instance.id}에서 기존 SSH 키들 제거 시작...")
        
        # 방법 1: Vast.ai 계정에 등록된 키들을 detach
        try:
            keys_response = self.client.show_ssh_keys()
            detached_count = 0
            
            for key_info in keys_response:
                key_id = key_info.get("id")
                if key_id:
                    try:
                        print(f"[SSH-CLEAN] Vast.ai 키 {key_id} detach 시도...")
                        self.client.detach_ssh(
                            instance_id=instance.id,
                            ssh_key_id=str(key_id)
                        )
                        detached_count += 1
                        print(f"[SSH-CLEAN] ✅ 키 {key_id} detach 완료")
                    except Exception as e:
                        print(f"[SSH-CLEAN] ⚠️  키 {key_id} detach 실패: {e}")
            
            print(f"[SSH-CLEAN] Vast.ai 키 detach 완료: {detached_count}개")
            
        except Exception as e:
            print(f"[SSH-CLEAN] Vast.ai 키 detach 중 오류: {e}")
        
        # 방법 2: SSH를 통해 authorized_keys 직접 정리 
        try:
            print("[SSH-CLEAN] SSH를 통한 authorized_keys 직접 정리 시도...")
            host, port = self.get_ssh_info(instance)
            ssh_key_path = os.getenv("SSH_KEY_PATH")
            
            if not ssh_key_path or not os.path.exists(ssh_key_path):
                print("[SSH-CLEAN] ⚠️  SSH 키 파일을 찾을 수 없음")
                return True  # detach만 했으니 부분 성공
            
            import paramiko
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                hostname=host,
                port=port,
                username="root",
                key_filename=ssh_key_path,
                timeout=10,
                look_for_keys=False
            )
            
            # authorized_keys를 백업하고 빈 파일로 초기화
            commands = [
                "cp ~/.ssh/authorized_keys ~/.ssh/authorized_keys.backup",
                "echo '' > ~/.ssh/authorized_keys",
                "chmod 600 ~/.ssh/authorized_keys"
            ]
            
            for cmd in commands:
                stdin, stdout, stderr = ssh.exec_command(cmd)
                stdout.read()  # 명령 완료 대기
            
            ssh.close()
            print("[SSH-CLEAN] ✅ authorized_keys 직접 정리 완료")
            return True
            
        except Exception as e:
            print(f"[SSH-CLEAN] SSH 직접 정리 실패: {e}")
            return True  # detach는 했으니 부분 성공
    
    def add_ssh_keys_to_instance(self, instance: VastInstance, clean_first: bool = False) -> bool:
        """계정의 모든 SSH 키를 인스턴스에 추가 (기존 키 보존)
        
        Args:
            instance: 대상 인스턴스
            clean_first: 기존 키들을 먼저 제거할지 여부 (기본값 False - 보존적 접근)
        """
        if not self.check_client() or not instance.id:
            return False
            
        print(f"[SSH-ADD] 인스턴스 {instance.id}에 SSH 키 추가 시작...")
        
        # 0. 기존 SSH 키들 정리 (선택사항 - 기본적으로 하지 않음)
        if clean_first:
            print("[SSH-CLEAN] 기존 SSH 키들 정리 중...")
            self._clean_instance_ssh_keys(instance)
        
        # 1. 현재 등록된 SSH 키 확인
        existing_keys = self._get_existing_ssh_keys(instance)
        print(f"[SSH-CHECK] 현재 등록된 키 {len(existing_keys)}개 확인")
        
        # 2. SSH 키 목록 조회
        ssh_keys = self.get_account_ssh_keys()
        if not ssh_keys:
            print("[SSH-ADD] 사용 가능한 SSH 키가 없습니다.")
            return False
        
        # 3. 새로 추가할 키 필터링 (중복 제거)
        new_keys = []
        for ssh_key in ssh_keys:
            key_parts = ssh_key.strip().split()
            if len(key_parts) >= 2:
                key_signature = key_parts[1][:20]  # 키의 고유 서명 일부
                if not any(key_signature in existing for existing in existing_keys):
                    new_keys.append(ssh_key)
                else:
                    print(f"[SSH-SKIP] 키가 이미 등록되어 있음 (서명: {key_signature})")
        
        if not new_keys:
            print("[SSH-ADD] 모든 키가 이미 등록되어 있습니다.")
            return True  # 이미 등록된 상태이므로 성공으로 간주
        
        print(f"[SSH-ADD] {len(new_keys)}개의 새 키를 추가합니다...")
        
        success_count = 0
        
        # 4. 새로운 SSH 키들만 인스턴스에 추가
        for i, ssh_key in enumerate(new_keys):
            try:
                if not ssh_key.strip():
                    continue
                    
                print(f"[SSH-ADD] SSH 키 {i+1}/{len(new_keys)} 추가 중...")
                
                # 방법 1: Vast.ai SDK attach_ssh 메서드 사용
                api_success = False
                try:
                    # attach_ssh는 항상 None을 반환하지만, 예외가 없으면 성공으로 간주
                    self.client.attach_ssh(
                        instance_id=instance.id,
                        ssh_key=ssh_key
                    )
                    
                    print(f"[SSH-ADD] ✅ SDK를 통해 SSH 키 {i+1} 추가 시도 완료")
                    api_success = True
                    success_count += 1
                        
                except Exception as api_e:
                    print(f"[SSH-ADD] SDK 방법 실패: {api_e}")
                
                # 방법 2: API 실패 시 SSH를 통해 직접 추가
                if not api_success:
                    print(f"[SSH-ADD] SSH를 통한 직접 키 추가 시도...")
                    if self._add_ssh_key_via_ssh(instance, ssh_key):
                        print(f"[SSH-ADD] ✅ SSH를 통해 키 {i+1} 추가 성공")
                        success_count += 1
                    else:
                        print(f"[SSH-ADD] ❌ SSH를 통한 키 {i+1} 추가 실패")
                    
            except Exception as e:
                print(f"[SSH-ADD] SSH 키 {i+1} 추가 중 예외: {e}")
        
        print(f"[SSH-ADD] 완료: {success_count}/{len(new_keys)}개 새 키 추가 성공")
        return success_count > 0 or len(new_keys) == 0  # 추가할 키가 없어도 성공

    def _add_ssh_key_via_ssh(self, instance: VastInstance, public_key: str) -> bool:
        """SSH를 통해 직접 authorized_keys에 키 추가 (대체 방법)"""
        try:
            host, port = self.get_ssh_info(instance)
            if not host or not port:
                return False
                
            # 현재 SSH 키로 연결
            ssh_key_path = os.getenv("SSH_KEY_PATH")
            if not ssh_key_path or not os.path.exists(ssh_key_path):
                return False
            
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            ssh.connect(
                hostname=host,
                port=port,
                username="root",
                key_filename=ssh_key_path,
                timeout=15,
                look_for_keys=False
            )
            
            # 키의 주요 부분 추출 (중복 체크용)
            key_parts = public_key.strip().split()
            if len(key_parts) < 2:
                ssh.close()
                return False
            
            key_type = key_parts[0]  # ssh-rsa, ssh-ed25519 등
            key_data = key_parts[1][:20]  # 키 데이터 일부
            
            # 중복 키 확인
            check_command = f'grep -q "{key_data}" ~/.ssh/authorized_keys 2>/dev/null'
            stdin, stdout, stderr = ssh.exec_command(check_command, timeout=10)
            exit_status = stdout.channel.recv_exit_status()
            
            if exit_status == 0:
                print(f"[SSH-DIRECT] 키가 이미 존재함 (중복 방지)")
                ssh.close()
                return True  # 이미 존재하므로 성공으로 간주
            
            # .ssh 디렉토리 생성 (만약 없다면)
            setup_command = 'mkdir -p ~/.ssh && chmod 700 ~/.ssh'
            ssh.exec_command(setup_command, timeout=10)
            
            # authorized_keys에 키 추가 (중복 방지하여)
            safe_key = public_key.replace('"', '\\"').replace('$', '\\$')
            add_command = f'echo "{safe_key}" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
            stdin, stdout, stderr = ssh.exec_command(add_command, timeout=10)
            exit_status = stdout.channel.recv_exit_status()
            
            ssh.close()
            
            if exit_status == 0:
                print(f"[SSH-DIRECT] ✅ SSH를 통해 키 추가 성공")
                return True
            else:
                error_output = stderr.read().decode().strip()
                print(f"[SSH-DIRECT] ❌ 키 추가 명령 실패: {error_output}")
                return False
            
        except Exception as e:
            print(f"[SSH-DIRECT] SSH를 통한 키 추가 실패: {e}")
            return False

    def _add_ssh_keys_to_instance(self, instance: VastInstance) -> bool:
        """인스턴스에 SSH 키 추가 (래퍼 메서드)"""
        print("[SSH-KEYS] 팀 SSH 키들을 인스턴스에 등록 중...")
        return self.add_ssh_keys_to_instance(instance)
