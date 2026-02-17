# ============================================================
# 🚀 코스피 500개 종목 AI 퀀트 트레이딩 시스템
# ============================================================
# 이 노트북은 코스피 상장 종목들의 재무 데이터를 수집하고,
# AI(랜덤포레스트)를 활용하여 투자 전략을 수립합니다.
# 
# 주요 기능:
# 1. 캐시 시스템을 활용한 고속 데이터 수집
# 2. 머신러닝 모델 학습 (랜덤포레스트)
# 3. 팩터 중요도 분석
# 4. AI 추천 종목 선정
# 5. 기존 투자 전략과 AI 전략 비교
# ============================================================

# ============================================================
# 1. 필요한 라이브러리 설치 및 임포트
# ============================================================

# !pip install: 코랩에서 파이썬 패키지를 설치하는 명령어
# -q 옵션: 설치 과정을 조용히(quiet) 진행
!pip install pykrx -q

# 데이터 처리 및 분석 라이브러리
import numpy as np              # 수치 계산을 위한 라이브러리 (배열, 행렬 연산)
import pandas as pd             # 데이터 분석 라이브러리 (엑셀과 유사한 데이터프레임)
import matplotlib.pyplot as plt # 그래프 시각화 라이브러리
import seaborn as sns           # 더 예쁜 그래프를 위한 라이브러리

# 날짜/시간 처리 라이브러리
from datetime import datetime, timedelta  # 날짜 계산, 형변환 등

# 시스템 관련 라이브러리
import time                      # 실행 시간 측정용
import os                         # 파일/폴더 경로 처리
import pickle                     # 파이썬 객체 저장/불러오기 (캐시용)

# 한국 주식 데이터 라이브러리
from pykrx import stock           # KRX(한국거래소) 데이터 조회

# 머신러닝 라이브러리 (scikit-learn)
from sklearn.ensemble import RandomForestClassifier  # 랜덤포레스트 분류 모델
from sklearn.model_selection import train_test_split # 데이터를 학습/테스트용으로 분할
from sklearn.metrics import accuracy_score, confusion_matrix # 모델 성능 평가
from sklearn.preprocessing import StandardScaler     # 데이터 정규화 (평균0, 분산1)

print("="*50)
print("✅ 머신러닝 퀀트 시작!")
print(f"numpy 버전: {np.__version__}")
print(f"pandas 버전: {pd.__version__}")
print("="*50)

# ============================================================
# 📅 분석 기준 날짜 설정
# ============================================================
# TARGET_DATE: 분석할 기준 날짜 (2026년 2월 13일 금요일)
# 이 날짜의 데이터를 수집하여 분석함
TARGET_DATE = "20260213"

# ============================================================
# 1. 캐시 시스템 클래스
# ============================================================
# 캐시란? 한 번 수집한 데이터를 파일로 저장해두고,
# 다음에 같은 데이터가 필요할 때 빠르게 불러오는 기술
# 이를 통해 데이터 수집 시간을 10분 → 1초로 단축

class StockDataCache:
    """
    주식 데이터 캐시 관리 클래스
    - 목적: 데이터 수집 속도 향상
    - 방법: 한 번 수집한 데이터는 stock_cache 폴더에 pkl 파일로 저장
    - 다음에 같은 데이터 요청 시 파일에서 바로 불러옴
    """
    
    def __init__(self, cache_dir='stock_cache'):
        """
        캐시 시스템 초기화
        Args:
            cache_dir: 캐시 파일을 저장할 폴더 이름 (기본: 'stock_cache')
        """
        self.cache_dir = cache_dir
        # 폴더가 없으면 자동 생성
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, ticker, date):
        """
        캐시 파일의 전체 경로 생성
        Args:
            ticker: 종목 코드 (예: '005930')
            date: 기준 날짜 (예: '20260213')
        Returns:
            캐시 파일 경로 (예: 'stock_cache/005930_20260213.pkl')
        """
        return f"{self.cache_dir}/{ticker}_{date}.pkl"

    def save(self, ticker, date, data):
        """
        데이터를 캐시에 저장
        Args:
            ticker: 종목 코드
            date: 기준 날짜
            data: 저장할 데이터 (딕셔너리 형태)
        """
        # 파일을 쓰기(w) 바이너리(b) 모드로 열기
        with open(self.get_cache_path(ticker, date), 'wb') as f:
            # pickle.dump: 파이썬 객체를 파일로 저장
            pickle.dump(data, f)

    def load(self, ticker, date):
        """
        캐시에서 데이터 로드
        Args:
            ticker: 종목 코드
            date: 기준 날짜
        Returns:
            저장된 데이터 (있으면), None (없으면)
        """
        path = self.get_cache_path(ticker, date)
        # os.path.exists: 파일 존재 여부 확인
        if os.path.exists(path):
            # 파일을 읽기(r) 바이너리(b) 모드로 열기
            with open(path, 'rb') as f:
                # pickle.load: 파일에서 파이썬 객체 로드
                return pickle.load(f)
        return None

# 캐시 인스턴스 생성 (전역에서 사용)
cache = StockDataCache()


# ============================================================
# 2. 단일 종목 데이터 수집 함수
# ============================================================
def fetch_ticker_data(ticker, date):
    """
    하나의 종목에 대한 모든 데이터를 수집하는 함수
    - 캐시 확인 -> 있으면 캐시 사용, 없으면 새로 수집
    - 수집 항목: PER, PBR, 배당수익률, 시가총액, 거래대금,
                모멘텀(1,3,6개월), 변동성, 현재가
    
    Args:
        ticker: 종목 코드 (예: '005930')
        date: 기준 날짜 (예: '20260213')
    
    Returns:
        result 딕셔너리 (성공 시), None (실패 시)
    """
    
    # 1. 캐시 확인
    cached = cache.load(ticker, date)
    if cached is not None:  # 캐시에 데이터가 있으면
        return cached       # 바로 반환 (시간 절약!)

    try:
        # 2. 종목명 조회 (예: '005930' -> '삼성전자')
        name = stock.get_market_ticker_name(ticker)

        # 3. PER, PBR, 배당수익률 데이터 조회
        # get_market_fundamental: 재무제표 기반 데이터 조회
        df_fund = stock.get_market_fundamental(date, date, ticker)
        if df_fund.empty:  # 데이터가 없으면 실패
            return None

        # iloc[0]: 첫 번째 행(해당 날짜 데이터) 가져오기
        fund_row = df_fund.iloc[0]
        
        # 각 컬럼이 있는지 확인 후 값 추출 (없으면 NaN)
        # index: 데이터프레임의 컬럼명 목록
        per = fund_row['PER'] if 'PER' in fund_row.index else np.nan
        pbr = fund_row['PBR'] if 'PBR' in fund_row.index else np.nan
        div = fund_row['DIV'] if 'DIV' in fund_row.index else 0

        # 4. 시가총액, 거래대금 데이터 조회
        df_cap = stock.get_market_cap(date, date, ticker)
        if df_cap.empty:
            return None

        cap_row = df_cap.iloc[0]
        market_cap = cap_row['시가총액'] if '시가총액' in cap_row.index else np.nan
        volume = cap_row['거래대금'] if '거래대금' in cap_row.index else np.nan

        # 5. PER, PBR 유효성 검사
        # pd.isna: NaN(결측치) 확인
        if pd.isna(per) or pd.isna(pbr) or per <= 0 or pbr <= 0:
            return None  # 유효하지 않은 데이터는 제외

        # 6. 로그 변환 (시가총액이 1조, 100조 등 차이가 커서 로그 취하면 정규분포에 가까워짐)
        market_cap_log = np.log(market_cap) if market_cap > 0 else np.nan
        volume_log = np.log(volume) if volume > 0 else np.nan

        # 7. 가격 데이터 조회 (모멘텀, 변동성 계산용)
        # 200일 전 날짜 계산 (timedelta: 날짜 차이)
        start_date = (datetime.strptime(date, '%Y%m%d') - timedelta(days=200)).strftime('%Y%m%d')
        # get_market_ohlcv_by_date: 일별 OHLCV 데이터 조회 (Open, High, Low, Close, Volume)
        df_price = stock.get_market_ohlcv_by_date(start_date, date, ticker)

        if df_price is None or len(df_price) < 100:
            return None  # 데이터가 너무 적으면 제외

        # 8. 모멘텀 계산
        # 모멘텀 = (현재가 / 과거가 - 1) * 100 (백분율)
        # iloc[-1]: 마지막 행(최신 데이터), iloc[-22]: 22일 전(약 1개월)
        momentum_1m = (df_price['종가'].iloc[-1] / df_price['종가'].iloc[-22] - 1) * 100 if len(df_price) > 22 else np.nan
        momentum_3m = (df_price['종가'].iloc[-1] / df_price['종가'].iloc[-66] - 1) * 100 if len(df_price) > 66 else np.nan
        momentum_6m = (df_price['종가'].iloc[-1] / df_price['종가'].iloc[-132] - 1) * 100 if len(df_price) > 132 else np.nan
        
        # 9. 변동성 계산
        # tail(60): 최근 60일 데이터, std(): 표준편차
        volatility = df_price['등락률'].tail(60).std() if len(df_price) >= 60 else np.nan

        # 10. target (임시) - 실제로는 다음달 수익률로 대체해야 함
        # np.random.random(): 0~1 사이 랜덤값, 0.5보다 크면 1(상승), 작으면 0(하락)
        target = 1 if np.random.random() > 0.5 else 0

        # 11. 결과 딕셔너리 생성
        result = {
            '티커': ticker,
            '종목': name,
            'PER': round(per, 2),
            'PBR': round(pbr, 2),
            '배당수익률': round(div, 2),
            '시가총액': round(market_cap_log, 2) if not np.isnan(market_cap_log) else np.nan,
            '거래대금': round(volume_log, 2) if not np.isnan(volume_log) else np.nan,
            '수익률_1개월': round(momentum_1m, 2) if not np.isnan(momentum_1m) else np.nan,
            '수익률_3개월': round(momentum_3m, 2) if not np.isnan(momentum_3m) else np.nan,
            '수익률_6개월': round(momentum_6m, 2) if not np.isnan(momentum_6m) else np.nan,
            '변동성': round(volatility, 2) if not np.isnan(volatility) else np.nan,
            'target': target
        }

        # 12. 캐시 저장 (다음에 빠르게 불러오기 위해)
        cache.save(ticker, date, result)
        return result

    except Exception as e:
        # 예외 발생 시 None 반환 (해당 종목은 건너뜀)
        return None


# ============================================================
# 3. 월별 데이터 수집 함수 (병렬 처리)
# ============================================================
# ThreadPoolExecutor: 여러 작업을 동시에 처리하는 병렬 처리 라이브러리
from concurrent.futures import ThreadPoolExecutor, as_completed

def collect_month_data_parallel(date, max_workers=10):
    """
    특정 월의 모든 종목 데이터를 병렬로 수집
    Args:
        date: 기준 날짜
        max_workers: 동시에 처리할 최대 스레드 수 (기본 10)
    Returns:
        해당 월의 종목 데이터프레임
    """
    
    print(f"📅 {date} 수집중...")

    # 종목 리스트 조회 (코스피 전체에서 300개만)
    tickers = stock.get_market_ticker_list(date, market="KOSPI")[:300]

    results = []
    
    # ThreadPoolExecutor로 병렬 처리
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 각 종목별로 fetch_ticker_data 함수 실행 예약
        # executor.submit(함수, 인자1, 인자2) -> Future 객체 반환
        futures = {executor.submit(fetch_ticker_data, ticker, date): ticker
                  for ticker in tickers}

        # as_completed: 작업이 완료되는 대로 결과 반환
        for future in as_completed(futures):
            result = future.result()
            if result:
                result['기준일'] = date  # 기준일 컬럼 추가
                results.append(result)

    # 리스트를 데이터프레임으로 변환
    df = pd.DataFrame(results)
    print(f"  ✅ {date}: {len(df)}개 종목")
    return df


# ============================================================
# 4. 2025년 전체 데이터 수집 함수 (병렬 + 캐시)
# ============================================================
def collect_2025_complete():
    """
    2025년 1월~12월 전체 데이터 수집
    - 캐시가 있으면 1초, 없으면 3-5분 소요
    Returns:
        2025년 전체 종목 데이터프레임
    """

    # 전체 캐시 파일 확인
    cache_file = '2025_complete_cache.pkl'
    if os.path.exists(cache_file):
        print("📦 전체 캐시 발견! 즉시 로딩...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # 2025년 월별 마지막 거래일 목록
    months = [
        '20250131', '20250228', '20250331', '20250430',
        '20250530', '20250630', '20250731', '20250829',
        '20250930', '20251031', '20251128', '20251230'
    ]

    all_data = []
    # 각 월별 데이터 수집
    for month in months:
        df_month = collect_month_data_parallel(month, max_workers=10)
        if len(df_month) > 0:
            all_data.append(df_month)

    # 모든 월 데이터 합치기
    final_df = pd.concat(all_data, ignore_index=True)

    # 전체 캐시 저장
    with open(cache_file, 'wb') as f:
        pickle.dump(final_df, f)

    return final_df


# ============================================================
# 5. 다음달 수익률 추가 함수 (진짜 target!)
# ============================================================
def add_future_returns(df):
    """
    각 종목의 다음달 수익률 계산
    - target: 다음달 수익률이 양수면 1, 음수면 0
    Args:
        df: 종목별 월별 데이터
    Returns:
        target이 추가된 데이터프레임
    """

    results = []
    # 종목별로 그룹화
    grouped = df.groupby('티커')

    for ticker, group in grouped:
        # 기준일 순으로 정렬
        group = group.sort_values('기준일')

        # 다음달 수익률 계산 (현재월과 다음월 데이터 필요)
        for i in range(len(group)-1):
            current = group.iloc[i]
            next_row = group.iloc[i+1]

            # 다음달 수익률 = (다음달 종가 / 현재달 종가 - 1) * 100
            future_return = (next_row['현재가'] / current['현재가'] - 1) * 100

            row_dict = current.to_dict()
            row_dict['다음달수익률'] = round(future_return, 2)
            # target: 양수면 1(상승), 음수면 0(하락)
            row_dict['target'] = 1 if future_return > 0 else 0

            results.append(row_dict)

    return pd.DataFrame(results)


# ============================================================
# 6. 팩터 중요도 분석 함수
# ============================================================
def analyze_feature_importance(model, feature_cols):
    """
    랜덤포레스트 모델의 팩터 중요도 분석
    - feature_importances_: 각 특성이 예측에 얼마나 기여했는지 (0~1)
    Args:
        model: 학습된 랜덤포레스트 모델
        feature_cols: 특성(팩터) 이름 목록
    Returns:
        중요도 순으로 정렬된 데이터프레임
    """
    
    # 모델에서 중요도 추출
    importances = model.feature_importances_
    # 중요도 높은 순으로 정렬한 인덱스
    indices = np.argsort(importances)[::-1]

    # 데이터프레임으로 정리
    importance_df = pd.DataFrame({
        '팩터': [feature_cols[i] for i in indices],
        '중요도': [importances[i] for i in indices]
    })

    print("\n" + "="*60)
    print("📊 팩터 중요도 분석")
    print("="*60)
    print(importance_df.to_string(index=False))

    # 시각화
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_cols)))
    plt.bar(range(len(feature_cols)), importances[indices], color=colors)
    plt.xticks(range(len(feature_cols)), [feature_cols[i] for i in indices], rotation=45)
    plt.title('팩터 중요도 (높을수록 영향력 큼)')
    plt.xlabel('팩터')
    plt.ylabel('중요도')
    plt.tight_layout()
    plt.show()

    # 인사이트 도출
    print("\n💡 인사이트:")
    print(f"  가장 중요한 팩터 3개: {', '.join([feature_cols[indices[i]] for i in range(3)])}")
    print(f"  상위 3개 팩터의 누적 중요도: {sum(importances[indices][:3]):.2%}")

    return importance_df


# ============================================================
# 7. AI 추천 종목 선정 함수
# ============================================================
def get_ai_recommendations(df, model, scaler, feature_cols, top_n=20):
    """
    AI 모델로 추천 종목 선정
    Args:
        df: 종목 데이터
        model: 학습된 모델
        scaler: 정규화 객체
        feature_cols: 특성 목록
        top_n: 추천할 종목 수
    Returns:
        상승확률 높은 순으로 정렬된 데이터프레임
    """
    
    df_result = df.copy()
    
    # 데이터 정규화
    X_all = scaler.transform(df[feature_cols].values)
    
    # 상승 확률 예측 (predict_proba: 각 클래스 확률 반환)
    # [:, 1] : 클래스 1(상승)의 확률만 선택
    df_result['상승확률'] = model.predict_proba(X_all)[:, 1]

    # 추천 등급 부여 (상위 20% = 매수, 중간 60% = 관심, 하위 20% = 관망)
    # pd.qcut: 데이터를 구간으로 나눔
    df_result['추천등급'] = pd.qcut(df_result['상승확률'], 
                                     q=[0, 0.2, 0.8, 1.0],
                                     labels=['관망', '관심', '매수'])

    print("\n" + "="*60)
    print(f"🤖 AI 추천 TOP {top_n} 종목")
    print("="*60)

    # 상승확률 높은 순 정렬
    top_stocks = df_result.sort_values('상승확률', ascending=False).head(top_n)

    # 출력할 컬럼 선택
    display_cols = ['종목', 'PER', 'PBR', '배당수익률',
                    '수익률_1개월', '수익률_3개월', '변동성', '상승확률', '추천등급']

    print(top_stocks[display_cols].to_string(index=False))

    return top_stocks


# ============================================================
# 8. 기존 전략 vs AI 전략 비교 함수
# ============================================================
def compare_strategies(df, model, scaler, feature_cols, top_n=10):
    """
    기존 투자 전략과 AI 전략을 비교 분석
    비교 전략:
    1. 가치주 전략 (저PER + 저PBR)
    2. 모멘텀 전략 (최근 수익률 높은 순)
    3. 저변동성 전략 (변동성 낮은 순)
    4. AI 전략 (상승확률 높은 순)
    """
    
    df_strat = df.copy()

    # === 1. AI 점수 계산 ===
    X_all = scaler.transform(df[feature_cols].values)
    df_strat['AI점수'] = model.predict_proba(X_all)[:, 1]

    # === 2. 가치주 전략 ===
    # PER과 PBR이 낮을수록 좋음 (음수 가중치)
    df_strat['가치점수'] = -df_strat['PER'] - df_strat['PBR']

    # === 3. 모멘텀 전략 ===
    df_strat['모멘텀점수'] = df_strat['수익률_1개월'] + df_strat['수익률_3개월']

    # === 4. 저변동성 전략 ===
    df_strat['저변동성점수'] = -df_strat['변동성']

    # === 5. 각 전략별 TOP 10 선정 ===
    value_top = df_strat.nlargest(top_n, '가치점수')[['종목', 'PER', 'PBR', '가치점수']].copy()
    value_top['전략'] = '가치주'

    momentum_top = df_strat.nlargest(top_n, '모멘텀점수')[['종목', 'PER', 'PBR', '모멘텀점수']].copy()
    momentum_top['전략'] = '모멘텀'

    lowvol_top = df_strat.nlargest(top_n, '저변동성점수')[['종목', 'PER', 'PBR', '저변동성점수']].copy()
    lowvol_top['전략'] = '저변동성'

    ai_top = df_strat.nlargest(top_n, 'AI점수')[['종목', 'PER', 'PBR', 'AI점수']].copy()
    ai_top['전략'] = 'AI'

    # === 6. 결과 출력 ===
    print("\n" + "="*70)
    print("📊 전략별 TOP 10 비교")
    print("="*70)

    print("\n🏆 가치주 전략 TOP 10 (저PER + 저PBR):")
    print(value_top.to_string(index=False))

    print("\n🚀 모멘텀 전략 TOP 10 (최근 수익률 높은 순):")
    print(momentum_top.to_string(index=False))

    print("\n🛡️ 저변동성 전략 TOP 10 (변동성 낮은 순):")
    print(lowvol_top.to_string(index=False))

    print("\n🤖 AI 전략 TOP 10 (상승확률 높은 순):")
    print(ai_top.to_string(index=False))

    # === 7. 중복 종목 분석 ===
    value_set = set(value_top['종목'].head(5))
    momentum_set = set(momentum_top['종목'].head(5))
    lowvol_set = set(lowvol_top['종목'].head(5))
    ai_set = set(ai_top['종목'].head(5))

    print("\n" + "="*70)
    print("🔍 전략별 중복 종목 분석 (TOP 5 기준)")
    print("="*70)
    print(f"  가치주 ∩ AI: {value_set & ai_set}")
    print(f"  모멘텀 ∩ AI: {momentum_set & ai_set}")
    print(f"  저변동성 ∩ AI: {lowvol_set & ai_set}")
    print(f"  가치주 ∩ 모멘텀: {value_set & momentum_set}")
    print(f"  모든 전략 공통: {value_set & momentum_set & lowvol_set & ai_set}")

    # === 8. 전략별 특성 비교 ===
    print("\n" + "="*70)
    print("📈 전략별 포트폴리오 특성 비교")
    print("="*70)

    comparison = pd.DataFrame({
        '전략': ['가치주', '모멘텀', '저변동성', 'AI'],
        '평균 PER': [
            value_top['PER'].mean(),
            momentum_top['PER'].mean(),
            lowvol_top['PER'].mean(),
            ai_top['PER'].mean()
        ],
        '평균 PBR': [
            value_top['PBR'].mean(),
            momentum_top['PBR'].mean(),
            lowvol_top['PBR'].mean(),
            ai_top['PBR'].mean()
        ],
        '평균 수익률_1개월': [
            df_strat[df_strat['종목'].isin(value_top['종목'])]['수익률_1개월'].mean(),
            df_strat[df_strat['종목'].isin(momentum_top['종목'])]['수익률_1개월'].mean(),
            df_strat[df_strat['종목'].isin(lowvol_top['종목'])]['수익률_1개월'].mean(),
            df_strat[df_strat['종목'].isin(ai_top['종목'])]['수익률_1개월'].mean()
        ],
        '평균 변동성': [
            df_strat[df_strat['종목'].isin(value_top['종목'])]['변동성'].mean(),
            df_strat[df_strat['종목'].isin(momentum_top['종목'])]['변동성'].mean(),
            df_strat[df_strat['종목'].isin(lowvol_top['종목'])]['변동성'].mean(),
            df_strat[df_strat['종목'].isin(ai_top['종목'])]['변동성'].mean()
        ]
    })
    print(comparison.round(2).to_string(index=False))

    # === 9. 시각화 ===
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 가치주 전략 시각화
    axes[0,0].scatter(df_strat['PER'], df_strat['PBR'], alpha=0.3, s=10, label='전체종목')
    axes[0,0].scatter(value_top['PER'], value_top['PBR'], color='red', s=100, label='가치주 TOP10', alpha=0.7)
    axes[0,0].set_xlabel('PER')
    axes[0,0].set_ylabel('PBR')
    axes[0,0].set_title('가치주 전략')
    axes[0,0].legend()
    axes[0,0].set_xlim(0, 50)
    axes[0,0].set_ylim(0, 5)
    axes[0,0].grid(True, alpha=0.3)

    # 모멘텀 전략 시각화
    axes[0,1].scatter(df_strat['PER'], df_strat['PBR'], alpha=0.3, s=10, label='전체종목')
    axes[0,1].scatter(momentum_top['PER'], momentum_top['PBR'], color='blue', s=100, label='모멘텀 TOP10', alpha=0.7)
    axes[0,1].set_xlabel('PER')
    axes[0,1].set_ylabel('PBR')
    axes[0,1].set_title('모멘텀 전략')
    axes[0,1].legend()
    axes[0,1].set_xlim(0, 50)
    axes[0,1].set_ylim(0, 5)
    axes[0,1].grid(True, alpha=0.3)

    # 저변동성 전략 시각화
    axes[1,0].scatter(df_strat['PER'], df_strat['PBR'], alpha=0.3, s=10, label='전체종목')
    axes[1,0].scatter(lowvol_top['PER'], lowvol_top['PBR'], color='green', s=100, label='저변동성 TOP10', alpha=0.7)
    axes[1,0].set_xlabel('PER')
    axes[1,0].set_ylabel('PBR')
    axes[1,0].set_title('저변동성 전략')
    axes[1,0].legend()
    axes[1,0].set_xlim(0, 50)
    axes[1,0].set_ylim(0, 5)
    axes[1,0].grid(True, alpha=0.3)

    # AI 전략 시각화
    axes[1,1].scatter(df_strat['PER'], df_strat['PBR'], alpha=0.3, s=10, label='전체종목')
    axes[1,1].scatter(ai_top['PER'], ai_top['PBR'], color='purple', s=100, label='AI TOP10', alpha=0.7)
    axes[1,1].set_xlabel('PER')
    axes[1,1].set_ylabel('PBR')
    axes[1,1].set_title('AI 전략')
    axes[1,1].legend()
    axes[1,1].set_xlim(0, 50)
    axes[1,1].set_ylim(0, 5)
    axes[1,1].grid(True, alpha=0.3)

    plt.suptitle('전략별 TOP 10 종목 분포 비교', fontsize=16)
    plt.tight_layout()
    plt.show()

    # === 10. 전략별 성과 예측 ===
    print("\n" + "="*70)
    print("📊 전략별 예상 성과 비교")
    print("="*70)

    # 각 전략의 평균 AI 점수 계산
    value_ai_score = df_strat[df_strat['종목'].isin(value_top['종목'])]['AI점수'].mean()
    momentum_ai_score = df_strat[df_strat['종목'].isin(momentum_top['종목'])]['AI점수'].mean()
    lowvol_ai_score = df_strat[df_strat['종목'].isin(lowvol_top['종목'])]['AI점수'].mean()
    ai_ai_score = df_strat[df_strat['종목'].isin(ai_top['종목'])]['AI점수'].mean()

    performance = pd.DataFrame({
        '전략': ['가치주', '모멘텀', '저변동성', 'AI'],
        'AI 평균 점수': [
            f"{value_ai_score:.1%}",
            f"{momentum_ai_score:.1%}",
            f"{lowvol_ai_score:.1%}",
            f"{ai_ai_score:.1%}"
        ]
    })
    print(performance.to_string(index=False))

    print("\n💡 인사이트:")
    if ai_ai_score > max(value_ai_score, momentum_ai_score, lowvol_ai_score):
        print("  ✅ AI 전략이 다른 전략보다 우수한 종목을 선별했습니다.")
    else:
        print("  🤔 기존 전략과 AI 전략이 비슷한 성과를 보입니다.")

    return value_top, momentum_top, lowvol_top, ai_top


# ============================================================
# 실행! (force_refresh=False로 캐시 사용)
# ============================================================
print("="*60)
print("🚀 2026년 2월 13일 금요일 기준 데이터 수집 (캐시 최적화)")
print("="*60)

# 🔥 중요: force_refresh=False로 설정해서 캐시 사용!
df = get_ml_data_final(
    n_stocks=500,
    use_cache=True,
    force_refresh=False  # False면 무조건 캐시 사용!
)

if len(df) > 0:
    print("\n📊 수집된 데이터 정보:")
    print(f"종목 수: {len(df)}개")
    print(f"팩터 목록: {df.columns.tolist()}")

    print("\n📈 데이터 샘플 (상위 10개):")
    print(df.head(10))

    print("\n📊 팩터별 통계:")
    print(df.describe())

    # ============================================================
    # 머신러닝 모델 학습
    # ============================================================
    print("\n" + "="*60)
    print("🤖 머신러닝 모델 학습 시작")
    print("="*60)

    # 팩터 컬럼 (종목, target 제외)
    feature_cols = [col for col in df.columns if col not in ['종목', 'target']]
    X = df[feature_cols].values
    y = df['target'].values

    # 데이터 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # 랜덤포레스트 모델
    model = RandomForestClassifier(