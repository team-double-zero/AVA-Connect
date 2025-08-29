"""Vast.ai helper 모듈: GPU 오퍼 검색 및 인스턴스 관리."""

import os
import warnings
from typing import Any, Dict, List, Optional, Union

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
        cpu_ram: 시스템 RAM 용량 (MB)
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
        self.gpu_ram: Optional[int] = offer_data.get("gpu_ram")  # MB
        self.gpu_frac: Optional[float] = offer_data.get("gpu_frac")
        self.dph_total: Optional[float] = offer_data.get("dph_total")
        self.dlperf_per_dph: Optional[float] = offer_data.get("dlperf_per_dphtotal")
        self.reliability: Optional[float] = offer_data.get("reliability")
        self.geolocation: Optional[str] = offer_data.get("geolocation")
        
        # 시스템 스펙
        self.cpu_cores: Optional[int] = offer_data.get("cpu_cores")
        self.cpu_ram: Optional[int] = offer_data.get("cpu_ram")  # MB
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
        
        # 실행 상태
        self.cur_state: Optional[str] = data.get("cur_state")
        self.intended_status: Optional[str] = data.get("intended_status")
        
        # 네트워크 정보
        self.ssh_host: Optional[str] = data.get("ssh_host")
        self.ssh_port: Optional[int] = data.get("ssh_port")
        self.public_ipaddr: Optional[str] = data.get("public_ipaddr")
        
        # 메타데이터
        self.reliability: Optional[float] = data.get("reliability2")
        self.geolocation: Optional[str] = data.get("geolocation")
        self.ports: Optional[Any] = data.get("ports")
        self.ssh: Optional[Any] = data.get("ssh")
    
    def __str__(self) -> str:
        """인스턴스 정보를 문자열로 반환."""
        return f"VastInstance(id={self.id}, gpu_name={self.gpu_name}, gpu_ram={self.gpu_ram}, gpu_frac={self.gpu_frac}, dph_total={self.dph_total}, cur_state={self.cur_state}, ssh_host={self.ssh_host}, ssh_port={self.ssh_port}, geolocation={self.geolocation})"
    
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
            weight_dlperf (float): dlperf_per_dphtotal 가중치
            weight_reliability (float): reliability 가중치
            weight_gpu_frac (float): gpu_frac 가중치
            reliability_scale (float): reliability 스케일링 계수
            gpu_frac_scale (float): gpu_frac 스케일링 계수
        """

        if not self.check_client(print_output=print_output):
            return None

        try:
            offers = self.client.search_offers(
                query=f"gpu_name~{gpu_model} rentable=true rented=false verified=true",
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


def run_function_tests() -> None:
    """각 public 함수별 테스트 실행."""
    from dotenv import load_dotenv
    load_dotenv()
    
    print("🧪 VastHelper 함수별 테스트")
    print("=" * 50)
    
    tests = {
        "1": {
            "name": "find_best_offer() - A100 GPU 검색",
            "func": "find_best_offer"
        },
        "2": {
            "name": "get_instances() - 인스턴스 목록 조회", 
            "func": "get_instances"
        }
    }
    
    # 메뉴 출력
    print("\n📋 테스트할 함수:")
    for key, test in tests.items():
        print(f"  {key}. {test['name']}")
    
    print("\n  0. 종료")
    
    while True:
        try:
            choice = input("\n선택하세요 (0-2): ").strip()
            
            if choice == "0":
                print("👋 테스트를 종료합니다.")
                break
                
            if choice not in tests:
                print("❌ 잘못된 선택입니다. 0-2 중에서 선택해주세요.")
                continue
                
            test = tests[choice]
            print(f"\n🎯 테스트: {test['name']}")
            print("-" * 60)
            
            if test["func"] == "find_best_offer":
                # VastHelper().find_best_offer() 테스트
                helper = VastHelper()
                result = helper.find_best_offer(
                    print_output=True,
                    gpu_model="A100",
                    min_vram_mb=40960,  # 40GB+
                    min_gpu_frac=0.5
                )
                if result:
                    print(f"\n✅ 결과: GPU ID {result.id} 추천됨")
                else:
                    print("\n❌ 조건에 맞는 GPU를 찾지 못했습니다.")
                    
            elif test["func"] == "get_instances":
                # VastHelper().get_instances() 테스트
                helper = VastHelper()
                instances = helper.get_instances()
                if instances:
                    print(f"✅ 총 {len(instances)}개의 인스턴스를 찾았습니다:")
                    for i, inst in enumerate(instances, 1):
                        print(f"  {i}. {inst}")
                else:
                    print("❌ 인스턴스를 찾을 수 없거나 조회에 실패했습니다.")
            
            print("\n" + "=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 사용자 요청으로 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            continue


if __name__ == "__main__":
    run_function_tests()