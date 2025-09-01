"""범용 재시도 데코레이터 모듈"""

import time
import functools
import logging
from typing import Callable, Type, Tuple, Optional, Any

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetryConfig:
    """재시도 설정 클래스"""
    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        max_delay: float = 60.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        ignore_exceptions: Tuple[Type[Exception], ...] = ()
    ):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff
        self.max_delay = max_delay
        self.exceptions = exceptions
        self.ignore_exceptions = ignore_exceptions

def retry_on_failure(
    config: Optional[RetryConfig] = None,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ignore_exceptions: Tuple[Type[Exception], ...] = (),
    log_attempts: bool = True,
    return_on_final_failure: Any = None
):
    """범용 재시도 데코레이터
    
    Args:
        config: RetryConfig 객체 (제공되면 다른 파라미터 무시)
        max_attempts: 최대 재시도 횟수 (기본값: 3)
        delay: 초기 대기 시간 (기본값: 1초)
        backoff: 대기 시간 증배 계수 (기본값: 2.0)
        max_delay: 최대 대기 시간 (기본값: 60초)
        exceptions: 재시도할 예외 타입들 (기본값: 모든 예외)
        ignore_exceptions: 재시도하지 않을 예외 타입들
        log_attempts: 재시도 과정 로그 출력 여부
        return_on_final_failure: 최종 실패 시 반환할 값 (None이면 예외 발생)
    """
    
    # config가 제공되면 해당 설정 사용
    if config:
        max_attempts = config.max_attempts
        delay = config.delay
        backoff = config.backoff
        max_delay = config.max_delay
        exceptions = config.exceptions
        ignore_exceptions = config.ignore_exceptions
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0 and log_attempts:
                        logger.info(f"✅ {func.__name__} 성공 (시도: {attempt + 1}/{max_attempts})")
                    return result
                    
                except ignore_exceptions as e:
                    # 무시할 예외는 바로 재발생
                    if log_attempts:
                        logger.info(f"❌ {func.__name__} - 무시할 예외 발생: {type(e).__name__}: {e}")
                    raise
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # 마지막 시도 실패
                        if log_attempts:
                            logger.error(f"❌ {func.__name__} 최종 실패 ({max_attempts}번 시도): {type(e).__name__}: {e}")
                        
                        if return_on_final_failure is not None:
                            return return_on_final_failure
                        else:
                            raise
                    else:
                        # 재시도 진행
                        if log_attempts:
                            logger.warning(f"🔄 {func.__name__} 실패 (시도: {attempt + 1}/{max_attempts}): {type(e).__name__}: {e}")
                            logger.info(f"⏳ {current_delay:.1f}초 후 재시도...")
                        
                        time.sleep(current_delay)
                        current_delay = min(current_delay * backoff, max_delay)
                        
        return wrapper
    return decorator

# 사전 정의된 재시도 설정들
NETWORK_RETRY = RetryConfig(
    max_attempts=5,
    delay=2.0,
    backoff=1.5,
    max_delay=30.0,
    exceptions=(ConnectionError, TimeoutError, OSError),
    ignore_exceptions=()
)

SSH_RETRY = RetryConfig(
    max_attempts=3,
    delay=5.0,
    backoff=2.0,
    max_delay=60.0,
    exceptions=(Exception,),  # SSH 관련 모든 예외
    ignore_exceptions=()
)

API_RETRY = RetryConfig(
    max_attempts=4,
    delay=1.0,
    backoff=2.0,
    max_delay=15.0,
    exceptions=(ConnectionError, TimeoutError),
    ignore_exceptions=()
)

INSTALLATION_RETRY = RetryConfig(
    max_attempts=2,
    delay=10.0,
    backoff=1.5,
    max_delay=30.0,
    exceptions=(RuntimeError,),
    ignore_exceptions=()
)

# 편의 함수들
def with_network_retry(func):
    """네트워크 관련 재시도 데코레이터"""
    return retry_on_failure(config=NETWORK_RETRY)(func)

def with_ssh_retry(func):
    """SSH 관련 재시도 데코레이터"""
    return retry_on_failure(config=SSH_RETRY)(func)

def with_api_retry(func):
    """API 관련 재시도 데코레이터"""
    return retry_on_failure(config=API_RETRY)(func)

def with_installation_retry(func):
    """설치 관련 재시도 데코레이터"""
    return retry_on_failure(config=INSTALLATION_RETRY)(func)

# 사용 예시
if __name__ == "__main__":
    import random
    
    @retry_on_failure(max_attempts=3, delay=0.5, log_attempts=True)
    def flaky_function():
        """테스트용 불안정 함수"""
        if random.random() < 0.7:
            raise ConnectionError("네트워크 연결 실패")
        return "성공!"
    
    @with_network_retry
    def network_operation():
        """네트워크 작업 예시"""
        if random.random() < 0.8:
            raise ConnectionError("일시적 네트워크 오류")
        return "네트워크 작업 완료"
    
    # 테스트
    try:
        result = flaky_function()
        print(f"결과: {result}")
    except:
        print("최종 실패")
    
    try:
        result = network_operation()
        print(f"네트워크 결과: {result}")
    except:
        print("네트워크 작업 최종 실패")
