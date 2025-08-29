import os
import warnings
from vastai_sdk import VastAI

POLL_INTERVAL = 5 # 초

class VastInstance:
    def __init__(self, data: dict):
        self.status = (data.get("actual_status") or data.get("status_msg") or data.get("status") or "").lower()
        self.id = data.get("id")
        self.gpu_name = data.get("gpu_name")
        self.gpu_ram = data.get("gpu_ram")  # MB
        self.gpu_frac = data.get("gpu_frac")
        self.dph_total = data.get("dph_total")
        self.dlperf_per_dph = data.get("dlperf_per_dph")
        self.cur_state = data.get("cur_state")
        self.intended_status = data.get("intended_status")
        self.ssh_host = data.get("ssh_host")
        self.ssh_port = data.get("ssh_port")
        self.public_ipaddr = data.get("public_ipaddr")
        self.reliability = data.get("reliability2")
        self.geolocation = data.get("geolocation")
        self.ports = data.get("ports")
        self.ssh = data.get("ssh")

    def __str__(self):
        return f"VastInstance(id={self.id}, gpu_name={self.gpu_name}, gpu_ram={self.gpu_ram}, gpu_frac={self.gpu_frac}, dph_total={self.dph_total}, dlperf_per_dph={self.dlperf_per_dph}, cur_state={self.cur_state}, intended_status={self.intended_status}, ssh_host={self.ssh_host}, ssh_port={self.ssh_port}, public_ipaddr={self.public_ipaddr}, reliability={self.reliability}, geolocation={self.geolocation}, ports={self.ports}, ssh={self.ssh})"

class VastHelper:
    """Vast.ai 헬퍼: 내부적으로 VastAI 클라이언트를 관리하고, 오퍼 검색/랭킹 제공"""

    def __init__(self, api_key: str | None = None):
        # 환경 변수에서 키를 읽되, 이 단계에서는 .env 로딩을 수행하지 않음
        self.api_key = api_key or os.getenv("VAST_API_KEY")
        self.client = None
        if not self.api_key:
            warnings.warn("VAST_API_KEY가 설정되지 않았습니다. VastHelper는 API 키가 필요합니다.")
            return
        try:
            self.client = VastAI(api_key=self.api_key)
        except Exception as exc:
            warnings.warn(f"VastAI 클라이언트 초기화 실패: {exc}")
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
    def calculate_metrics(offer):
        """가성비 지표들을 계산"""
        dph = offer.get("dph_total", float("inf"))
        vram_gb = offer.get("gpu_ram", 0) / 1024  # MB to GB
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
        m,
        weight_dlperf: float = 0.6,
        weight_reliability: float = 0.2,
        weight_gpu_frac: float = 0.2,
        reliability_scale: float = 20.0,
        gpu_frac_scale: float = 20.0,
    ):
        """종합 점수 계산 (딥러닝 성능, 신뢰도, GPU 전용도 고려)"""
        dlperf_term = m['dlperf_per_dph'] * weight_dlperf
        reliability_term = m['reliability'] * reliability_scale * weight_reliability
        gpu_frac_term = m['gpu_frac'] * gpu_frac_scale * weight_gpu_frac
        return dlperf_term + reliability_term + gpu_frac_term

    def find_best_offer(
        self,
        print_output: bool = False,
        gpu_model: str = "A100",
        min_vram_mb: int = 0,
        min_gpu_frac: float = 0,
        weight_dlperf: float = 0.6,
        weight_reliability: float = 0.2,
        weight_gpu_frac: float = 0.2,
        reliability_scale: float = 20.0,
        gpu_frac_scale: float = 20.0,
    ):
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
                limit=200,
            )
        except Exception as exc:
            if print_output:
                print(f"오퍼 조회 중 오류가 발생했습니다: {exc}")
            return None

        want = [
            o for o in offers
            if (
                "gpu_name" in o
                and gpu_model.lower() in o["gpu_name"].lower()
                and o.get("gpu_ram", 0) >= min_vram_mb
                and o.get("gpu_frac", 1.0) >= min_gpu_frac
            )
        ]

        metrics = [self.calculate_metrics(o) for o in want]

        best_dl_perf = sorted(metrics, key=lambda x: (-x['dlperf_per_dph'], x['dph_total']))
        best_cost_per_vram = sorted(metrics, key=lambda x: x['cost_per_vram_gb'])
        best_price = sorted(metrics, key=lambda x: x['dph_total'])

        if print_output:
            vram_gb = min_vram_mb / 1024
            print(f"🚀 {gpu_model} {vram_gb:.0f}GB+ GPU 추천 순위\n" + "="*80)

            print("\n📊 1. 딥러닝 가성비 TOP 5 (성능/가격 기준)")
            print(f"{'ID':<8} {'GPU':<16} {'VRAM':<7} {'Price':<8} {'DL성능/가격':<12} {'신뢰도':<6} {'GPU비율':<8}")
            print("-" * 75)
            for i, m in enumerate(best_dl_perf[:5], 1):
                o = m['offer']
                print(f"{o.get('id',''):<8} {o.get('gpu_name',''):<16} {m['vram_gb']:<7.1f} ${m['dph_total']:<7.3f} {m['dlperf_per_dph']:<12.2f} {m['reliability']:<6.2f} {m['gpu_frac']:<8.2f}")

            print(f"\n💰 2. VRAM 가성비 TOP 5 ($/GB 기준)")
            print(f"{'ID':<8} {'GPU':<16} {'VRAM':<7} {'Price':<8} {'$/GB·h':<8} {'신뢰도':<6} {'GPU비율':<8}")
            print("-" * 70)
            for i, m in enumerate(best_cost_per_vram[:5], 1):
                o = m['offer']
                print(f"{o.get('id',''):<8} {o.get('gpu_name',''):<16} {m['vram_gb']:<7.1f} ${m['dph_total']:<7.3f} {m['cost_per_vram_gb']:<7.4f} {m['reliability']:<6.2f} {m['gpu_frac']:<8.2f}")

            print(f"\n💸 3. 최저가 TOP 5")
            print(f"{'ID':<8} {'GPU':<16} {'VRAM':<7} {'Price':<8} {'DL성능/가격':<12} {'신뢰도':<6}")
            print("-" * 68)
            for i, m in enumerate(best_price[:5], 1):
                o = m['offer']
                print(f"{o.get('id',''):<8} {o.get('gpu_name',''):<16} {m['vram_gb']:<7.1f} ${m['dph_total']:<7.3f} {m['dlperf_per_dph']:<12.2f} {m['reliability']:<6.2f}")

        best_overall = sorted(
            metrics,
            key=lambda m: self.comprehensive_score(
                m,
                weight_dlperf=weight_dlperf,
                weight_reliability=weight_reliability,
                weight_gpu_frac=weight_gpu_frac,
                reliability_scale=reliability_scale,
                gpu_frac_scale=gpu_frac_scale,
            ),
            reverse=True,
        )

        if print_output:
            print(f"\n⭐ 최종 추천 GPU")
            print("="*50)

        if best_overall:
            top = best_overall[0]
            o = top['offer']

            if print_output:
                print(f"ID: {o.get('id', '')}")
                print(f"GPU: {o.get('gpu_name', '')}")
                print(f"VRAM: {top['vram_gb']:.1f}GB")
                print(f"가격: ${top['dph_total']:.3f}/시간")
                print(f"DL 성능/가격: {top['dlperf_per_dph']:.2f}")
                print(f"신뢰도: {top['reliability']:.2f}")
                print(f"GPU 전용도: {top['gpu_frac']:.2f}")
                print(f"VRAM 당 비용: ${top['cost_per_vram_gb']:.4f}/GB·h")
                print(f"위치: {o.get('geolocation', 'N/A')}")

            return top
        else:
            if print_output:
                print("조건에 맞는 GPU를 찾을 수 없습니다.")
            return None

    def get_instances(self) -> list[VastInstance]:
        """인스턴스 목록 조회"""
        if not self.check_client():
            return None
        return [VastInstance(inst) for inst in self.client.show_instances()]


# 하위 호환: 기존 함수형 API 유지 (내부적으로 클래스 사용)
def find_best_gpu(*args, **kwargs):
    helper = VastHelper()
    return helper.find_best_offer(*args, **kwargs)


if __name__ == "__main__":
    VastHelper().find_best_offer(print_output=True, min_gpu_frac=0.5)