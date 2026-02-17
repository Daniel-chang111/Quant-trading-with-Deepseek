# ============================================================
# 🚀 코스피 퀀트 트레이딩 시스템 v2.0
# ============================================================
# 📌 이 프로그램이 하는 일:
#   1. 코스피 상위 500개 종목의 주식 데이터를 3년치 수집
#   2. 수집한 데이터로 AI 모델을 학습시킴
#      (어떤 종목이 다음달에 오를지 패턴을 배움)
#   3. 2026년 1월 기준으로 오를 것 같은 종목 TOP 20 선정
#   4. 2026년 2월에 실제로 얼마나 올랐는지 성과 확인
# ============================================================


# ============================================================
# 0. 필요한 도구(패키지) 설치 및 불러오기
# ============================================================

# !pip install : 파이썬에서 외부 도구를 설치하는 명령어
# -q 옵션 : 설치 과정 메시지를 간략하게 출력 (quiet 모드)
!pip install pykrx pandas numpy scikit-learn lightgbm matplotlib seaborn -q

# ── 기본 수학/데이터 처리 도구 ────────────────────────────
import numpy as np          # 숫자 계산 전문 도구 (평균, 표준편차 등)
import pandas as pd         # 표(데이터프레임) 처리 전문 도구 (엑셀과 비슷)
import matplotlib.pyplot as plt  # 그래프 그리는 도구
import seaborn as sns       # 예쁜 그래프 그리는 도구 (matplotlib 보완)

# ── 날짜/시간 처리 도구 ───────────────────────────────────
from datetime import datetime, timedelta
# datetime : 날짜를 다루는 도구 (예: '20250131' → 날짜 객체로 변환)
# timedelta : 날짜 계산 도구 (예: 오늘 - 200일 = 200일 전 날짜)

# ── 주식 데이터 수집 도구 ─────────────────────────────────
from pykrx import stock
# pykrx : 한국 주식 데이터(KRX)를 무료로 가져오는 파이썬 라이브러리
# stock : pykrx 안에서 주식 관련 함수들의 묶음

# ── 기타 유틸리티 도구 ────────────────────────────────────
import time      # 시간 측정, 잠깐 멈추기(sleep) 등에 사용
import os        # 폴더 만들기, 파일 존재 확인 등 파일시스템 관련
import pickle    # 파이썬 데이터를 파일로 저장하고 불러오는 도구
import warnings  # 경고 메시지 제어 도구
from concurrent.futures import ThreadPoolExecutor, as_completed
# ThreadPoolExecutor : 여러 작업을 동시에 실행하는 도구 (병렬 처리)
# as_completed : 병렬 작업이 완료되는 순서대로 결과를 받아오는 도구

# ── AI 모델 관련 도구 ─────────────────────────────────────
from sklearn.ensemble import (
    RandomForestClassifier,      # 랜덤포레스트 모델 (여러 결정나무의 다수결)
    GradientBoostingClassifier,  # 그래디언트부스팅 모델 (실수를 반복 보완)
    VotingClassifier             # 여러 모델을 묶어서 투표하게 만드는 도구
)
from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀 (현재 미사용, 향후 확장용)
from sklearn.model_selection import TimeSeriesSplit  # 시계열 데이터 분할 도구
from sklearn.metrics import (
    accuracy_score,          # 정확도 계산 (맞은 비율)
    classification_report,   # 상세 성능 보고서 출력
    roc_auc_score            # AUC 점수 계산 (0.5=동전던지기, 1.0=완벽)
)
from sklearn.preprocessing import StandardScaler
# StandardScaler : 데이터 정규화 도구
# 예) PER은 1~100 범위, RSI는 0~100 범위 → 모두 비슷한 범위로 통일

import lightgbm as lgb
# LightGBM : Microsoft가 만든 고속 머신러닝 모델
# 대용량 데이터를 빠르게 처리하는 데 특화

warnings.filterwarnings('ignore')  # 불필요한 경고 메시지 숨기기
print("✅ 패키지 로드 완료")


# ============================================================
# 📦 캐시(Cache) 시스템
# ============================================================
# 캐시란? : 한 번 가져온 데이터를 파일로 저장해두는 것
# 왜 필요? : 인터넷으로 주식 데이터를 매번 다시 받으면 시간이 오래 걸림
#            저장해두면 다음에는 파일에서 바로 읽어서 훨씬 빠름
# 비유: 도서관에서 책을 빌려 집에 가져다 두면 다음에는 바로 읽을 수 있음

class StockDataCache:
    def __init__(self, cache_dir='stock_cache_v2'):
        # cache_dir : 캐시 파일을 저장할 폴더 이름
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        # os.makedirs : 폴더가 없으면 새로 만들기
        # exist_ok=True : 이미 폴더가 있어도 오류 없이 넘어가기

    def get_cache_path(self, ticker, date):
        # 캐시 파일 경로를 만드는 함수
        # 예) ticker='005930', date='20250131'
        #     → 'stock_cache_v2/005930_20250131.pkl'
        return f"{self.cache_dir}/{ticker}_{date}.pkl"

    def save(self, ticker, date, data):
        # 데이터를 파일로 저장하는 함수
        # 'wb' : write binary (이진 파일로 쓰기)
        with open(self.get_cache_path(ticker, date), 'wb') as f:
            pickle.dump(data, f)  # pickle.dump : 파이썬 데이터 → 파일로 저장

    def load(self, ticker, date):
        # 저장된 캐시 파일을 불러오는 함수
        # 파일이 없으면 None(없음)을 반환
        path = self.get_cache_path(ticker, date)
        if os.path.exists(path):           # 파일이 존재하면
            with open(path, 'rb') as f:    # 'rb' : read binary (이진 파일 읽기)
                return pickle.load(f)      # 파일 → 파이썬 데이터로 불러오기
        return None                        # 파일이 없으면 None 반환

# 캐시 객체 생성 (이후 코드에서 cache.save(), cache.load() 형태로 사용)
cache = StockDataCache()


# ============================================================
# 📅 유효 거래일 탐색 함수
# ============================================================
# 왜 필요? : 주식시장은 주말과 공휴일에 쉬어서 데이터가 없음
#            예) 1월 31일이 일요일이면 → 1월 30일(금요일) 데이터를 대신 사용
def find_valid_date(target_date, max_days_back=5):
    """
    입력한 날짜에 데이터가 없으면 최대 5일 전까지 거슬러 올라가며
    실제 거래일을 찾아주는 함수

    target_date  : 찾고 싶은 날짜 (예: '20250131')
    max_days_back: 최대 몇 일 전까지 찾을지 (기본값: 5일)
    """
    date_obj = datetime.strptime(target_date, '%Y%m%d')
    # datetime.strptime : 문자열 '20250131' → 날짜 객체로 변환
    # '%Y%m%d' : 연도4자리+월2자리+일2자리 형식 지정

    for i in range(max_days_back + 1):  # 0, 1, 2, 3, 4, 5 순서로 시도
        check_date = (date_obj - timedelta(days=i)).strftime('%Y%m%d')
        # timedelta(days=i) : i일을 빼서 이전 날짜 계산
        # .strftime('%Y%m%d') : 날짜 객체 → '20250131' 형식 문자열로 변환
        try:
            tickers = stock.get_market_ticker_list(check_date, market="KOSPI")
            # 해당 날짜의 코스피 종목 리스트 요청
            # 거래일이면 종목 목록이 오고, 휴장일이면 빈 리스트가 옴
            if tickers and len(tickers) > 0:   # 종목이 1개 이상 있으면 → 거래일!
                if i > 0:  # 날짜가 보정된 경우에만 알림 출력
                    print(f"  📅 {target_date} → {check_date}로 보정")
                return check_date  # 유효한 거래일 반환
        except:
            continue  # 오류가 나도 다음 날짜로 계속 시도
    return None  # 5일 다 시도해도 없으면 None 반환


# ============================================================
# 📊 시총 상위 N개 종목 조회 함수
# ============================================================
# 왜 시총 기준? : 종목코드 순서로 뽑으면 소형주가 많이 포함됨
#                시가총액(시총) 상위 종목이 데이터도 풍부하고
#                실제 투자도 가능한 종목들임
def get_top_tickers_by_market_cap(date, top_n=500):
    """
    특정 날짜 기준으로 시가총액 상위 N개 종목의 코드를 반환하는 함수

    date  : 기준 날짜 (예: '20250131')
    top_n : 상위 몇 개를 가져올지 (기본값: 500개)
    """
    try:
        df_cap = stock.get_market_cap(date, market="KOSPI")
        # 코스피 전체 종목의 시가총액 데이터 가져오기
        # df_cap : 종목코드별 시가총액이 담긴 표(데이터프레임)

        if df_cap is not None and not df_cap.empty:
            df_sorted = df_cap.sort_values('시가총액', ascending=False)
            # ascending=False : 큰 것부터 작은 순서로 정렬 (내림차순)
            tickers = df_sorted.index[:top_n].tolist()
            # .index : 행 인덱스 (여기서는 종목코드)
            # [:top_n] : 상위 top_n개만 선택
            # .tolist() : 리스트 형태로 변환
            print(f"  📊 시총 기준 상위 {len(tickers)}개 선정")
            return tickers

    except Exception as e:
        print(f"  ⚠️ 시총 조회 실패({e}), 기본 리스트 사용")

    # 시총 조회 실패 시 대안: 기본 종목 리스트에서 앞에서 top_n개 사용
    all_tickers = stock.get_market_ticker_list(date, market="KOSPI")
    return all_tickers[:top_n] if all_tickers else []


# ============================================================
# 🔍 단일 종목 데이터 수집 함수 (핵심 함수!)
# ============================================================
# 이 함수가 하는 일:
#   종목 하나에 대해 12가지 특성값을 계산해서 딕셔너리로 반환
#   특성 = AI 모델이 학습할 때 사용하는 재료들
#   (PER, PBR, 모멘텀, RSI 등 주식 분석에 쓰이는 수치들)
def fetch_ticker_data_v2(ticker, date):
    """
    특정 종목의 특성 데이터를 수집하는 함수

    ticker : 종목코드 (예: '005930' = 삼성전자)
    date   : 기준 날짜 (예: '20250131')
    반환값 : 12가지 특성값이 담긴 딕셔너리, 실패 시 None
    """

    # 먼저 캐시에 저장된 데이터가 있는지 확인
    cached = cache.load(ticker, date)
    if cached is not None:
        return cached  # 캐시에 있으면 바로 반환 (API 호출 생략)

    try:
        # 종목코드 → 종목명 변환 (예: '005930' → '삼성전자')
        name = stock.get_market_ticker_name(ticker)

        # ── 📊 펀더멘털 데이터 수집 (PER, PBR, 배당수익률) ──────
        # 펀더멘털 = 기업의 재무적 기본 가치를 나타내는 수치들
        df_fund = stock.get_market_fundamental(date, date, ticker)
        # get_market_fundamental : 특정 날짜의 재무 지표를 가져오는 함수
        # 날짜를 같은 값으로 두 번 쓰는 이유 : (시작일, 종료일) 형식이라서

        if df_fund is None or df_fund.empty:
            return None  # 데이터 없으면 이 종목 건너뜀

        fund_row = df_fund.iloc[0]  # 첫 번째 행 선택 (1일치 데이터이므로 1행)
        cols = fund_row.index.tolist()  # 컬럼 이름 목록 확인

        # PER (주가수익비율): 주가 / 주당순이익
        # 낮을수록 저평가 (예: PER 5 = 이익 대비 5배 가격)
        per = fund_row['PER'] if 'PER' in cols else np.nan

        # PBR (주가순자산비율): 주가 / 주당순자산
        # 1 미만이면 장부가치보다 싸게 거래됨 (저평가 신호)
        pbr = fund_row['PBR'] if 'PBR' in cols else np.nan

        # DIV (배당수익률): 배당금 / 주가 × 100 (%)
        # 높을수록 배당을 많이 줌 (예: DIV 5% = 100만원 투자 시 5만원 배당)
        div_yield = fund_row['DIV'] if 'DIV' in cols else np.nan

        # PER 또는 PBR이 없거나 0 이하면 분석 불가 → 건너뜀
        # np.nan : 숫자가 아님(Not a Number), 데이터 없음을 표현
        # pd.isna : 값이 nan인지 확인
        if pd.isna(per) or pd.isna(pbr) or per <= 0 or pbr <= 0:
            return None

        # ── 💰 시가총액 수집 ──────────────────────────────────
        # 시가총액 = 주가 × 총 발행 주식 수 = 회사의 총 가치
        market_cap = np.nan
        try:
            df_cap = stock.get_market_cap(date, date, ticker)
            if df_cap is not None and not df_cap.empty:
                market_cap = df_cap['시가총액'].iloc[0] / 1e8
                # 1e8 = 1억 → 억원 단위로 변환 (숫자가 너무 크면 학습에 방해됨)
        except:
            pass  # 시총 조회 실패해도 나머지 데이터는 계속 수집

        # ── 📈 가격/거래량 데이터 수집 (약 200일치) ──────────
        start_date = (datetime.strptime(date, '%Y%m%d') - timedelta(days=280)).strftime('%Y%m%d')
        # 280일 전부터 데이터를 가져오는 이유:
        # 주말/공휴일 제외하면 실제 거래일은 약 200일
        # 안전하게 280일로 설정해서 200 거래일치 확보

        df_price = stock.get_market_ohlcv_by_date(start_date, date, ticker)
        # get_market_ohlcv_by_date : 기간별 OHLCV 데이터 반환
        # OHLCV = Open(시가), High(고가), Low(저가), Close(종가), Volume(거래량)

        if df_price is None or len(df_price) < 100:
            return None  # 가격 데이터가 100일치 미만이면 분석 불가

        # 자주 쓸 데이터를 변수로 미리 저장
        closes  = df_price['종가']    # 매일의 종가(마감 가격) 시리즈
        volumes = df_price['거래량']  # 매일의 거래량 시리즈
        highs   = df_price['고가']    # 매일의 고가 시리즈
        lows    = df_price['저가']    # 매일의 저가 시리즈

        # ── 🚀 모멘텀 계산 ────────────────────────────────────
        # 모멘텀 = 일정 기간 동안의 주가 변화율 (%)
        # 양수 = 올랐음, 음수 = 떨어졌음
        # .iloc[-1] : 마지막(가장 최근) 값
        # .iloc[-22] : 22번째 전 값 (영업일 기준 약 1개월 전)

        momentum_1m = (closes.iloc[-1] / closes.iloc[-22]  - 1) * 100  # 1개월(22일) 수익률
        momentum_3m = (closes.iloc[-1] / closes.iloc[-66]  - 1) * 100  # 3개월(66일) 수익률
        momentum_6m = (closes.iloc[-1] / closes.iloc[-132] - 1) * 100 if len(df_price) > 132 else 0
        # 6개월(132일) 수익률, 데이터가 132일 미만이면 0으로 처리

        # ── 📉 변동성 계산 ────────────────────────────────────
        # 변동성 = 주가가 얼마나 들쭉날쭉한지 나타내는 수치
        # .std() : 표준편차 계산 (값이 클수록 주가 변동이 심함)
        # 최근 60일의 등락률(전일 대비 변화율) 표준편차
        volatility = df_price['등락률'].tail(60).std()
        # .tail(60) : 마지막 60개 데이터만 선택

        # ── 📦 거래량 변화율 계산 ─────────────────────────────
        # 거래량 급증 = 세력이 들어왔거나 뉴스가 있다는 신호
        vol_recent = volumes.tail(5).mean()   # 최근 5일 평균 거래량
        vol_avg20  = volumes.tail(20).mean()  # 최근 20일 평균 거래량
        vol_change = (vol_recent / vol_avg20 - 1) * 100 if vol_avg20 > 0 else 0
        # 최근 5일이 20일 평균 대비 몇 % 늘었는지
        # 양수 = 거래량 증가, 음수 = 거래량 감소

        # ── 💹 RSI 계산 (상대강도지수) ────────────────────────
        # RSI = 최근 14일간 상승 강도 / (상승 강도 + 하락 강도) × 100
        # 0~100 사이 값, 70 초과 = 과매수(너무 많이 올라 조정 가능),
        #                30 미만 = 과매도(너무 많이 떨어져 반등 가능)
        delta = closes.diff()  # 하루하루의 가격 변화량 계산 (오늘 - 어제)
        gain  = delta.clip(lower=0).tail(14).mean()   # 상승한 날만의 평균
        loss  = (-delta.clip(upper=0)).tail(14).mean() # 하락한 날만의 평균 (양수로 변환)
        # .clip(lower=0) : 0 미만 값은 0으로 (하락한 날은 0 처리)
        # .clip(upper=0) : 0 초과 값은 0으로 (상승한 날은 0 처리)
        rsi = 100 - (100 / (1 + gain / loss)) if loss != 0 else 50
        # loss가 0이면 (14일 내내 상승) RSI = 100에 가까우나 나누기 오류 방지로 50 설정

        # ── 📐 볼린저밴드 위치 계산 ───────────────────────────
        # 볼린저밴드 = 이동평균선 ± 2 × 표준편차로 만든 상·하단 밴드
        # 현재가가 밴드 내 어디에 위치하는지 0~1 사이 값으로 표현
        # 0 = 하단 밴드 (많이 떨어진 상태), 1 = 상단 밴드 (많이 오른 상태)
        ma20  = closes.tail(20).mean()  # 20일 이동평균선
        std20 = closes.tail(20).std()   # 20일 표준편차
        bb_pos = (closes.iloc[-1] - (ma20 - 2*std20)) / (4*std20) if std20 > 0 else 0.5
        # 분자: 현재가 - 하단밴드
        # 분모: 상단밴드 - 하단밴드 = 4 × 표준편차
        bb_pos = max(0, min(1, bb_pos))  # 0~1 범위 벗어나면 강제로 0 또는 1로 조정

        # ── 📏 52주 고저 대비 현재가 위치 ────────────────────
        # 52주(1년) 최고가와 최저가 사이에서 현재가가 어디 있는지 (%)
        # 0% = 52주 최저가, 100% = 52주 최고가
        high_52w = closes.tail(252).max() if len(closes) >= 252 else closes.max()
        low_52w  = closes.tail(252).min() if len(closes) >= 252 else closes.min()
        # 252 = 1년 영업일 수 (주5일 × 52주)
        pos_52w = (closes.iloc[-1] - low_52w) / (high_52w - low_52w) * 100 if (high_52w - low_52w) > 0 else 50

        # ── 결과를 딕셔너리로 정리 ────────────────────────────
        # 딕셔너리 = {키: 값} 형태의 데이터 구조
        result = {
            '티커':       ticker,   # 종목코드 (예: '005930')
            '종목':       name,     # 종목명 (예: '삼성전자')

            # 밸류에이션 지표 (기업 가치 대비 주가가 싼지 비싼지)
            'PER':        round(per, 2),    # 주가수익비율 (낮을수록 저평가)
            'PBR':        round(pbr, 2),    # 주가순자산비율 (낮을수록 저평가)
            '배당수익률': round(div_yield, 2) if not pd.isna(div_yield) else 0.0,

            # 규모 지표
            '시가총액':   round(market_cap, 0) if not pd.isna(market_cap) else np.nan,  # 억원

            # 모멘텀 지표 (주가 추세)
            '모멘텀_1m':  round(momentum_1m, 2),  # 1개월 수익률 (%)
            '모멘텀_3m':  round(momentum_3m, 2),  # 3개월 수익률 (%)
            '모멘텀_6m':  round(momentum_6m, 2),  # 6개월 수익률 (%)

            # 리스크/기술적 지표
            '변동성':     round(volatility, 2),   # 가격 변동 정도
            '거래량변화': round(vol_change, 2),   # 거래량 증감 (%)
            'RSI':        round(rsi, 2),           # 상대강도지수 (0~100)
            '볼린저위치': round(bb_pos, 4),        # 볼린저밴드 내 위치 (0~1)
            '52주위치':   round(pos_52w, 2),       # 52주 고저 대비 위치 (%)

            # 현재 주가 (다음달 수익률 계산에 사용)
            '현재가':     closes.iloc[-1],
        }

        cache.save(ticker, date, result)  # 결과를 캐시 파일로 저장
        return result

    except Exception as e:
        return None  # 어떤 오류가 나도 None 반환 (이 종목은 건너뜀)


# ============================================================
# 📅 월별 데이터 수집 함수 (병렬 처리 포함)
# ============================================================
# 병렬 처리란? : 여러 종목을 동시에 수집하는 것
# 비유: 1명이 500개 카드를 혼자 뒤집는 것 vs 3명이 나눠서 동시에 뒤집는 것
# max_workers=3 : 3개 작업을 동시에 실행 (너무 많으면 서버 차단 위험)
def collect_month_data_v2(date, top_n=500, max_workers=3):
    """
    특정 월의 시총 상위 500개 종목 데이터를 수집하는 함수

    date       : 수집 기준 날짜 (월말 거래일)
    top_n      : 수집할 종목 수
    max_workers: 동시에 몇 개 종목을 병렬로 수집할지
    """
    print(f"📅 {date} 수집 시작...")

    # 시총 기준 상위 종목 리스트 가져오기
    tickers = get_top_tickers_by_market_cap(date, top_n)
    if not tickers:
        print(f"  ⚠️ {date}: 종목 없음")
        return pd.DataFrame()  # 빈 데이터프레임 반환

    print(f"  📋 수집 대상: {len(tickers)}개")
    results = []  # 수집 결과를 담을 빈 리스트

    # 병렬 처리로 여러 종목 동시 수집
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.submit : 각 종목 수집 작업을 병렬 작업 대기열에 등록
        futures = {executor.submit(fetch_ticker_data_v2, t, date): t for t in tickers}
        # futures : {작업객체: 종목코드} 형태의 딕셔너리

        for i, future in enumerate(as_completed(futures)):
            # as_completed : 완료된 작업부터 순서대로 결과 받기
            # enumerate : 순번(i)과 값(future)을 함께 제공
            res = future.result()  # 완료된 작업의 결과 가져오기
            if res is not None:
                res['기준일'] = date  # 기준 날짜 컬럼 추가
                results.append(res)  # 결과 리스트에 추가

            # 50개마다 1초 쉬기 (서버 과부하 방지 - Rate Limit 보호)
            # KRX 서버에 너무 많은 요청을 빠르게 보내면 차단될 수 있음
            if i % 50 == 49:
                time.sleep(1)

    df = pd.DataFrame(results)  # 리스트 → 데이터프레임(표) 변환
    print(f"  ✅ 성공: {len(df)}개 / 전체: {len(tickers)}개")
    return df


# ============================================================
# 📊 개선 1: 다년도 데이터 수집 (2023~2025년 3개년)
# ============================================================
# 왜 3개년? : AI 모델이 다양한 시장 상황(상승장, 하락장, 횡보장)을
#             모두 경험해야 더 잘 예측할 수 있음
#             1년치(3,486개) → 3년치(약 10,000개+)로 학습 데이터 3배 확대
def collect_multi_year_data(start_year=2023, end_year=2025, top_n=500):
    """
    여러 연도의 월말 데이터를 수집하는 함수

    start_year : 시작 연도 (기본값: 2023)
    end_year   : 종료 연도 (기본값: 2025)
    top_n      : 종목 수 (기본값: 500)
    """

    # 각 연도의 월말 기준일 목록 생성
    month_targets = []
    for year in range(start_year, end_year + 1):
        month_ends = [
            f'{year}0131',  # 1월 말
            f'{year}0228',  # 2월 말
            f'{year}0331',  # 3월 말
            f'{year}0430',  # 4월 말
            f'{year}0530',  # 5월 말
            f'{year}0630',  # 6월 말
            f'{year}0731',  # 7월 말
            f'{year}0829',  # 8월 말 (8/31이 주말인 경우 대비)
            f'{year}0930',  # 9월 말
            f'{year}1031',  # 10월 말
            f'{year}1128',  # 11월 말 (11/30이 주말인 경우 대비)
            f'{year}1230',  # 12월 말 (12/31이 주말인 경우 대비)
        ]
        # 윤년 처리: 2024년은 2월이 29일까지 있음
        if year == 2024:
            month_ends[1] = '20240229'
        month_targets.extend(month_ends)  # 전체 목록에 추가

    print(f"📊 {start_year}~{end_year}년 실제 거래일 확인중 ({len(month_targets)}개월)...")

    # 각 월말 날짜를 실제 거래일로 보정
    valid_months = []
    for m in month_targets:
        valid = find_valid_date(m)  # 실제 거래일 탐색
        if valid:
            valid_months.append(valid)
        else:
            print(f"  ❌ {m}: 유효 거래일 없음")

    print(f"\n✅ 수집 대상: {len(valid_months)}개월\n")

    all_data = []       # 모든 월의 데이터를 담을 리스트
    start_time = time.time()  # 시작 시간 기록 (소요 시간 계산용)

    for i, month in enumerate(valid_months, 1):  # 1부터 시작하는 순번
        print(f"\n[{i}/{len(valid_months)}] ", end="")
        df_month = collect_month_data_v2(month, top_n=top_n)  # 해당 월 데이터 수집

        if df_month is not None and not df_month.empty:
            all_data.append(df_month)

        time.sleep(2)  # 월별 수집 사이에 2초 대기 (서버 부하 분산)

        # 남은 시간 예측 계산
        elapsed   = time.time() - start_time          # 경과 시간 (초)
        remaining = (elapsed / i) * (len(valid_months) - i)  # 예상 남은 시간
        print(f"  ⏱️ 경과: {elapsed:.0f}초 / 잔여: {remaining:.0f}초")

    if all_data:
        # 모든 월의 데이터를 하나의 큰 표로 합치기
        final_df = pd.concat(all_data, ignore_index=True)
        # pd.concat : 여러 데이터프레임을 세로로 이어 붙이기
        # ignore_index=True : 인덱스를 0부터 새로 부여
        print(f"\n✅ 수집 완료: {len(final_df)}개 샘플, {final_df['기준일'].nunique()}개월")
        # .nunique() : 고유값 개수 (중복 제외한 월 수)
        return final_df
    else:
        print("❌ 수집 실패")
        return pd.DataFrame()  # 빈 데이터프레임 반환


# ============================================================
# 🏷️ 레이블(정답) 생성 함수
# ============================================================
# AI 모델은 "문제(특성)"와 "정답(레이블)"을 같이 줘야 학습 가능
# 여기서 정답 = 다음달에 주가가 올랐는지(1) 내렸는지(0)
#
# 예시:
#   1월 말 삼성전자 가격: 70,000원
#   2월 말 삼성전자 가격: 73,500원
#   다음달수익률 = (73,500 / 70,000 - 1) × 100 = +5%
#   target = 1 (올랐으므로)
def add_future_returns(df):
    """
    각 종목의 다음달 수익률을 계산하고 상승/하락 레이블을 붙이는 함수

    df : 월별 종목 데이터가 담긴 데이터프레임
    반환값 : 다음달수익률과 target(0 또는 1) 컬럼이 추가된 데이터프레임
    """
    if df.empty:
        return pd.DataFrame()

    results = []

    # 종목별로 묶어서 처리 (같은 종목의 월별 데이터를 순서대로 비교)
    for ticker, group in df.groupby('티커'):
        group = group.sort_values('기준일').reset_index(drop=True)
        # sort_values('기준일') : 날짜 순으로 정렬 (오래된 것 → 최근 것)
        # reset_index(drop=True) : 인덱스를 0, 1, 2, ... 로 재설정

        if len(group) < 2:
            continue  # 최소 2개월치 데이터가 있어야 수익률 계산 가능

        for i in range(len(group) - 1):  # 마지막 월 제외 (다음달이 없으므로)
            row        = group.iloc[i].to_dict()    # i번째 월 데이터를 딕셔너리로
            next_price = group.iloc[i + 1]['현재가']  # i+1번째 월(다음달) 가격

            # 수익률 계산: (다음달 가격 / 이번달 가격 - 1) × 100
            ret = (next_price / row['현재가'] - 1) * 100
            row['다음달수익률'] = round(ret, 2)
            row['target'] = 1 if ret > 0 else 0  # 올랐으면 1, 내렸으면 0
            results.append(row)

    df_result = pd.DataFrame(results)
    print(f"✅ 학습 샘플: {len(df_result)}개 / 종목 수: {df_result['티커'].nunique()}개")

    # 월별 데이터 분포 확인 (상승 종목 비율이 편향되지 않은지 점검)
    monthly = df_result.groupby('기준일')['target'].agg(['count', 'mean'])
    monthly.columns = ['샘플수', '상승비율']
    monthly['상승비율'] = monthly['상승비율'].apply(lambda x: f"{x:.1%}")
    # lambda x: f"{x:.1%}" : 소수를 퍼센트 문자열로 변환 (예: 0.543 → '54.3%')
    print(f"\n📈 월별 학습 데이터:")
    print(monthly.to_string())
    return df_result


# ============================================================
# 🤖 AI 모델에 입력할 특성(Feature) 목록 정의
# ============================================================
# FEATURE_COLS : AI가 종목을 평가할 때 사용하는 12가지 기준
# 사람으로 치면 입사 지원자를 평가할 때 보는 항목들 (학점, 경력, 자격증 등)
FEATURE_COLS = [
    'PER', 'PBR', '배당수익률',           # 밸류에이션 (기업 가치 관련)
    '시가총액',                             # 규모 (대형주 vs 소형주)
    '모멘텀_1m', '모멘텀_3m', '모멘텀_6m', # 모멘텀 (주가 추세)
    '변동성', '거래량변화',                 # 리스크 (얼마나 위험한지)
    'RSI', '볼린저위치', '52주위치',        # 기술적 지표 (차트 분석)
]


# ============================================================
# 🏗️ 앙상블 모델 생성 함수
# ============================================================
# 앙상블 = 여러 AI 모델의 예측을 합쳐서 최종 결정
# 소프트 보팅 = 각 모델의 확률을 평균내서 결정
# (예: RF 70%, GBM 65%, LightGBM 45% → 평균 60% → 상승 예측)
def build_ensemble_model():
    """RF + GBM + LightGBM 세 모델을 합친 앙상블 모델 생성"""

    # 모델 1: 랜덤포레스트
    rf = RandomForestClassifier(
        n_estimators=200,      # 결정 나무 200그루 사용
        max_depth=8,           # 나무 깊이 최대 8단계 (너무 깊으면 과적합)
        min_samples_split=10,  # 가지를 나누려면 최소 10개 샘플 필요
        random_state=42,       # 결과 재현을 위한 난수 시드 고정 (42는 관례적 값)
        n_jobs=-1              # 모든 CPU 코어를 사용해서 빠르게 학습
    )

    # 모델 2: 그래디언트 부스팅
    gbm = GradientBoostingClassifier(
        n_estimators=150,    # 나무 150그루 순서대로 학습
        max_depth=4,         # 나무 깊이 최대 4단계 (얕게 해서 과적합 방지)
        learning_rate=0.05,  # 학습률 (한 번에 얼마나 배울지, 낮을수록 신중하게 학습)
        random_state=42
    )

    # 모델 3: LightGBM (Microsoft의 고속 부스팅)
    lgbm = lgb.LGBMClassifier(
        n_estimators=200,   # 나무 200그루
        max_depth=6,        # 나무 깊이 최대 6단계
        learning_rate=0.05, # 학습률
        num_leaves=31,      # 나무의 잎(말단 노드) 최대 31개
        random_state=42,
        n_jobs=-1,
        verbose=-1          # 학습 과정 출력 끄기
    )

    # 세 모델을 소프트 보팅으로 묶기
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gbm', gbm), ('lgbm', lgbm)],
        # estimators : (이름, 모델) 쌍의 리스트
        voting='soft',  # 'soft' = 확률 평균, 'hard' = 다수결
        n_jobs=-1       # 세 모델을 병렬로 학습
    )
    return ensemble


# ============================================================
# 🧪 개선 3+4: 시계열 교차검증 + 앙상블 모델 학습
# ============================================================
# 시계열 교차검증(Walk-Forward Validation)이란?
#
# ❌ 기존 랜덤 분할의 문제:
#    미래 데이터로 과거를 예측하는 상황이 발생 (데이터 누수)
#    예) 2025년 6월 데이터로 학습 → 2024년 1월을 예측 (현실에서 불가능!)
#
# ✅ 시계열 분할:
#    항상 과거로만 학습하고 미래로만 검증
#    예) Fold1: 2023년 1~12월로 학습 → 2024년 1월 검증
#        Fold2: 2023년 1월~2024년 1월로 학습 → 2024년 2월 검증
#        ...실제 투자와 동일한 조건으로 검증!
def train_with_timeseries_cv(df_train):
    """
    시계열 교차검증으로 앙상블 모델을 검증하고
    전체 데이터로 최종 모델을 학습하는 함수

    df_train : 레이블이 포함된 학습 데이터프레임
    반환값   : (최종 모델, 스케일러, 특성 목록) 튜플
    """

    # 결측치(NaN)가 있는 행 제거
    df = df_train.dropna(subset=FEATURE_COLS).copy()

    # 날짜순으로 정렬 (시계열 분할을 위해 필수)
    df    = df.sort_values('기준일').reset_index(drop=True)
    dates = sorted(df['기준일'].unique())  # 고유한 날짜 목록 (정렬)

    print(f"\n{'='*60}")
    print(f"🤖 앙상블 모델 시계열 교차검증")
    print(f"{'='*60}")
    print(f"📊 총 샘플: {len(df)}개 / {len(dates)}개월")
    print(f"📋 사용 특성 ({len(FEATURE_COLS)}개): {FEATURE_COLS}\n")

    # 검증 폴드 수 결정
    # min(6, len(dates)-12) : 최대 6번 검증, 최소 12개월 학습 데이터 확보
    n_splits  = min(6, len(dates) - 12)
    fold_accs = []  # 각 폴드의 정확도를 저장할 리스트
    fold_aucs = []  # 각 폴드의 AUC 점수를 저장할 리스트

    # Walk-Forward 검증 루프
    for fold in range(n_splits):
        # split_idx : 학습/검증 경계 인덱스
        split_idx   = len(dates) - n_splits + fold
        train_dates = dates[:split_idx]    # 경계 이전 = 학습 기간
        val_dates   = [dates[split_idx]]   # 경계 시점 = 검증 기간 (1개월)

        # 학습/검증 데이터 분리
        df_tr  = df[df['기준일'].isin(train_dates)]  # 학습 데이터
        df_val = df[df['기준일'].isin(val_dates)]    # 검증 데이터

        if len(df_tr) < 100 or len(df_val) < 10:
            continue  # 데이터 너무 적으면 건너뜀

        # 특성(X)과 레이블(y) 분리
        X_tr  = df_tr[FEATURE_COLS].values   # 학습 특성 행렬
        y_tr  = df_tr['target'].values        # 학습 정답 벡터
        X_val = df_val[FEATURE_COLS].values  # 검증 특성 행렬
        y_val = df_val['target'].values       # 검증 정답 벡터

        # 정규화: 각 특성의 범위를 비슷하게 통일
        # fit_transform : 학습 데이터로 정규화 기준을 정하고 적용
        # transform : 동일한 기준으로 검증 데이터에 적용 (기준은 새로 정하지 않음!)
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr)   # 학습 데이터 정규화
        X_val_sc = scaler.transform(X_val)       # 검증 데이터 정규화

        # 앙상블 모델 학습
        model = build_ensemble_model()
        model.fit(X_tr_sc, y_tr)  # 모델이 패턴을 학습하는 단계

        # 성능 평가
        y_pred = model.predict(X_val_sc)          # 상승/하락 예측 (0 또는 1)
        y_prob = model.predict_proba(X_val_sc)[:, 1]  # 상승 확률 (0.0~1.0)
        # predict_proba : [[하락확률, 상승확률], ...] 형태
        # [:, 1] : 모든 행의 1번 열(상승확률)만 선택

        acc = accuracy_score(y_val, y_pred)  # 정확도 = 맞춘 비율
        auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.5
        # AUC : 0.5=동전던지기 수준, 0.6=어느 정도 예측력, 0.7+=좋음

        fold_accs.append(acc)
        fold_aucs.append(auc)
        print(f"  Fold {fold+1} | 학습: {train_dates[0]}~{train_dates[-1]} | "
              f"검증: {val_dates[0]} | ACC: {acc:.3f} | AUC: {auc:.3f}")

    # 교차검증 결과 요약
    print(f"\n{'='*60}")
    print(f"📊 교차검증 결과")
    print(f"  평균 정확도: {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
    # ± 뒤의 값(표준편차)이 작을수록 폴드마다 성능이 일정함 (안정적)
    print(f"  평균 AUC:   {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")
    print(f"{'='*60}\n")

    # ── 전체 데이터로 최종 모델 학습 ─────────────────────────
    # 교차검증은 성능 평가용, 실제 예측에는 전체 데이터로 학습한 모델 사용
    print("🤖 전체 데이터로 최종 모델 학습중...")
    X_all    = df[FEATURE_COLS].values
    y_all    = df['target'].values
    scaler   = StandardScaler()
    X_all_sc = scaler.fit_transform(X_all)

    final_model = build_ensemble_model()
    final_model.fit(X_all_sc, y_all)
    print("✅ 최종 모델 학습 완료")

    # ── 특성 중요도 출력 (RF 기준) ───────────────────────────
    # 특성 중요도 = AI가 판단할 때 어떤 특성을 가장 많이 참고했는지
    try:
        rf_model   = final_model.estimators_[0]  # 앙상블에서 RF 모델만 추출
        importance = pd.DataFrame({
            '특성':   FEATURE_COLS,
            '중요도': rf_model.feature_importances_
            # feature_importances_ : 각 특성이 예측에 기여한 비율 (합계=1.0)
        }).sort_values('중요도', ascending=False)
        print(f"\n🏆 특성 중요도 (RandomForest 기준):")
        print(importance.to_string(index=False))
    except:
        pass

    return final_model, scaler, FEATURE_COLS


# ============================================================
# 💰 2026년 1월 투자 종목 선정 함수
# ============================================================
def select_stocks_jan2026_v2(model, scaler, features, date='20260130', top_n=500):
    """
    학습된 AI 모델로 2026년 1월 기준 상승 가능성 높은 종목 TOP 20 선정

    model    : 학습된 앙상블 모델
    scaler   : 학습 시 사용한 정규화 기준 (같은 기준 적용 필수!)
    features : 사용할 특성 목록
    date     : 기준 날짜 (기본값: 2026년 1월 30일)
    top_n    : 분석할 종목 수
    """
    valid_date = find_valid_date(date)  # 실제 거래일 탐색
    if not valid_date:
        print("❌ 유효 거래일 없음")
        return pd.DataFrame()

    print(f"\n📅 {valid_date} 데이터 수집중...")
    tickers = get_top_tickers_by_market_cap(valid_date, top_n)
    if not tickers:
        return pd.DataFrame()

    # 종목별 데이터 수집 (병렬 처리 없이 순차적으로)
    stock_data = []
    for ticker in tickers:
        res = fetch_ticker_data_v2(ticker, valid_date)
        if res is not None:
            res['기준일'] = valid_date
            stock_data.append(res)
        time.sleep(0.02)  # 종목당 0.02초 대기 (서버 부하 방지)

    df_jan = pd.DataFrame(stock_data)
    print(f"✅ {len(df_jan)}개 종목 수집 완료")

    if df_jan.empty:
        return pd.DataFrame()

    # 결측치 있는 종목 제거 (특성값이 하나라도 없으면 AI 점수 계산 불가)
    df_jan_clean = df_jan.dropna(subset=features)
    print(f"✅ 결측치 제거 후: {len(df_jan_clean)}개")

    # AI 점수 계산
    X_jan = scaler.transform(df_jan_clean[features].values)
    # 학습 때와 동일한 정규화 기준(scaler) 적용

    df_jan_clean = df_jan_clean.copy()
    df_jan_clean['AI_점수'] = model.predict_proba(X_jan)[:, 1]
    # predict_proba : 각 종목의 상승 확률 계산
    # [:, 1] : 상승 확률 값만 추출 (0.0~1.0)
    # AI_점수가 높을수록 다음달에 오를 가능성이 높다고 AI가 판단한 것

    # AI 점수 기준 상위 20개 종목 선정
    top_20 = df_jan_clean.nlargest(20, 'AI_점수')
    # nlargest(20, 'AI_점수') : AI_점수 기준 상위 20개 행 선택

    print("\n" + "="*80)
    print(f"💰 {valid_date} 투자 선정 종목 TOP 20")
    print("="*80)
    display_cols = ['종목', 'PER', 'PBR', '배당수익률', '모멘텀_1m', 'RSI', 'AI_점수']
    print(top_20[display_cols].to_string(index=False))
    return top_20


# ============================================================
# 📈 2026년 2월 성과 확인 함수
# ============================================================
def check_feb_results_v2(portfolio_jan, date='20260213'):
    """
    1월에 선정한 종목들의 2월 수익률을 확인하는 함수

    portfolio_jan : 1월에 선정된 포트폴리오 데이터프레임
    date          : 성과 확인 날짜 (기본값: 2026년 2월 13일)
    """
    valid_date = find_valid_date(date)
    if not valid_date:
        print("❌ 유효 거래일 없음")
        return pd.DataFrame()

    print(f"\n📅 {valid_date} 성과 확인중...")
    results = []

    for _, row in portfolio_jan.iterrows():
        # iterrows() : 데이터프레임을 한 행씩 순서대로 읽기
        ticker    = row['티커']
        buy_price = row['현재가']  # 1월 말 매수가

        try:
            # 2월의 종가 데이터 가져오기
            df_feb = stock.get_market_ohlcv_by_date(valid_date, valid_date, ticker)
            if df_feb is None or df_feb.empty:
                continue

            feb_price = df_feb['종가'].iloc[0]  # 2월 종가
            ret = (feb_price / buy_price - 1) * 100  # 수익률 계산

            results.append({
                '종목':      row['종목'],
                '매수가':    buy_price,
                '현재가':    feb_price,
                '수익률(%)': round(ret, 2),
                'AI_점수':   round(row['AI_점수'], 3),
            })
        except:
            continue

    df_res = pd.DataFrame(results).sort_values('수익률(%)', ascending=False)
    # 수익률 기준 내림차순 정렬 (높은 수익률 종목이 위로)

    print("\n" + "="*60)
    print("📈 2월 성과 결과")
    print("="*60)
    print(df_res.to_string(index=False))

    # 최종 통계 출력
    print(f"\n{'='*60}")
    print(f"✅ 분석 종목:   {len(df_res)}개")
    print(f"✅ 평균 수익률: {df_res['수익률(%)'].mean():.2f}%")
    print(f"✅ 중앙값:      {df_res['수익률(%)'].median():.2f}%")
    # 중앙값 = 수익률을 크기순으로 줄 세웠을 때 가운데 값
    # 평균보다 이상치(극단값)의 영향을 덜 받음
    print(f"✅ 상승 종목:   {len(df_res[df_res['수익률(%)'] > 0])}개")
    print(f"✅ 하락 종목:   {len(df_res[df_res['수익률(%)'] < 0])}개")
    print(f"✅ 승률:        {len(df_res[df_res['수익률(%)']>0]) / len(df_res) * 100:.1f}%")
    return df_res


# ============================================================
# 📊 결과 시각화 함수
# ============================================================
def visualize_results(df_res, df_train, portfolio_jan):
    """
    4개의 그래프로 결과를 시각화하는 함수

    df_res       : 2월 성과 결과
    df_train     : 학습에 사용한 데이터
    portfolio_jan: 1월에 선정된 포트폴리오
    """
    # 2행 2열로 4개의 그래프 공간 만들기
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # figsize=(14, 10) : 가로 14인치, 세로 10인치 크기
    fig.suptitle('퀀트 트레이딩 시스템 v2.0 결과', fontsize=14, fontweight='bold')

    # ── 그래프 1: 종목별 수익률 막대 차트 (왼쪽 위) ──────────
    ax1 = axes[0, 0]
    colors = ['#e74c3c' if r < 0 else '#2ecc71' for r in df_res['수익률(%)']]
    # 수익률이 음수면 빨간색(#e74c3c), 양수면 초록색(#2ecc71)
    ax1.barh(df_res['종목'], df_res['수익률(%)'], color=colors)
    # barh : 가로 막대 차트 (horizontal bar)
    ax1.axvline(x=0, color='black', linewidth=0.8)  # 0% 기준선 표시
    ax1.set_xlabel('수익률 (%)')
    ax1.set_title('📈 종목별 수익률')
    ax1.set_xlim(df_res['수익률(%)'].min() * 1.2, df_res['수익률(%)'].max() * 1.2)
    # 그래프 x축 범위를 최소/최대값보다 20% 여유 있게 설정

    # ── 그래프 2: AI 점수 vs 실제 수익률 산점도 (오른쪽 위) ──
    ax2 = axes[0, 1]
    # portfolio_jan의 AI_점수와 df_res의 수익률을 같은 종목 기준으로 매칭
    ax2.scatter(
        portfolio_jan['AI_점수'],
        df_res.set_index('종목').reindex(portfolio_jan['종목'])['수익률(%)'],
        alpha=0.7,        # 점의 투명도 (0=투명, 1=불투명)
        s=80,             # 점의 크기
        color='#3498db'   # 파란색
    )
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)  # 0% 기준선
    ax2.set_xlabel('AI 점수')
    ax2.set_ylabel('수익률 (%)')
    ax2.set_title('🤖 AI 점수 vs 실제 수익률')
    # AI 점수가 높은 종목이 실제로 수익률도 높은지 확인하는 그래프

    # ── 그래프 3: 월별 시장 상승 비율 (왼쪽 아래) ────────────
    ax3 = axes[1, 0]
    monthly_up = df_train.groupby('기준일')['target'].mean() * 100
    # 월별로 묶어서 target의 평균 계산 → 상승 종목 비율
    ax3.plot(range(len(monthly_up)), monthly_up.values, marker='o', color='#9b59b6')
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)  # 50% 기준선
    ax3.set_xticks(range(len(monthly_up)))
    ax3.set_xticklabels([d[2:6] for d in monthly_up.index], rotation=45, fontsize=7)
    # d[2:6] : '20250131' → '2501' (연월만 표시)
    # rotation=45 : 레이블을 45도 기울여서 겹치지 않게
    ax3.set_ylabel('상승 종목 비율 (%)')
    ax3.set_title('📊 월별 시장 상승 비율')

    # ── 그래프 4: PBR 분포 히스토그램 (오른쪽 아래) ──────────
    ax4 = axes[1, 1]
    up_pbr   = df_train[df_train['target'] == 1]['PBR'].clip(0, 5)  # 상승 종목의 PBR
    down_pbr = df_train[df_train['target'] == 0]['PBR'].clip(0, 5)  # 하락 종목의 PBR
    # .clip(0, 5) : PBR을 0~5 범위로 제한 (극단적 이상치 제거)
    ax4.hist(up_pbr,   bins=30, alpha=0.6, color='#2ecc71', label='상승')
    ax4.hist(down_pbr, bins=30, alpha=0.6, color='#e74c3c', label='하락')
    # bins=30 : 막대를 30개 구간으로 나누기
    # alpha=0.6 : 60% 불투명 (겹치는 부분이 보이도록)
    ax4.set_xlabel('PBR')
    ax4.set_ylabel('종목 수')
    ax4.set_title('📊 PBR 분포 (상승 vs 하락)')
    ax4.legend()  # 범례 표시

    plt.tight_layout()  # 그래프 간격 자동 조정
    plt.savefig('quant_results_v2.png', dpi=150, bbox_inches='tight')
    # dpi=150 : 해상도 설정 (높을수록 선명, 파일 크기도 커짐)
    # bbox_inches='tight' : 여백 최소화
    plt.show()
    print("✅ 시각화 저장: quant_results_v2.png")


# ============================================================
# 🚀 메인 실행 블록
# ============================================================
# if __name__ == '__main__' :
#   이 파일을 직접 실행할 때만 아래 코드가 실행됨
#   다른 파일에서 import할 때는 실행되지 않음
#   구글 코랩에서는 이 조건 없이도 실행되지만 코드 구조상 좋은 관습
if __name__ == '__main__':
    print("\n" + "🚀"*30)
    print("🚀 퀀트 트레이딩 시스템 v2.0 시작")
    print("🚀"*30 + "\n")

    # ── Step 1: 2023~2025년 3개년 데이터 수집 ─────────────────
    print("=" * 60)
    print("Step 1. 2023~2025년 데이터 수집 (시총 상위 500개)")
    print("=" * 60)
    df_raw = collect_multi_year_data(start_year=2023, end_year=2025, top_n=500)
    # 예상 소요시간: 약 3~4시간 (36개월 × 500종목)
    # 캐시가 있으면 훨씬 빠름

    if df_raw.empty:
        print("❌ 데이터 수집 실패. 종료.")
    else:
        # ── Step 2: 레이블(다음달 수익률) 생성 ────────────────
        print("\n" + "=" * 60)
        print("Step 2. 학습 레이블 생성 (다음달 수익률)")
        print("=" * 60)
        df_train = add_future_returns(df_raw)

        if df_train.empty:
            print("❌ 학습 데이터 생성 실패. 종료.")
        else:
            # ── Step 3: 앙상블 모델 + 시계열 교차검증 ────────
            print("\n" + "=" * 60)
            print("Step 3. 앙상블 모델 학습 (시계열 교차검증)")
            print("=" * 60)
            model, scaler, features = train_with_timeseries_cv(df_train)
            # 반환값 3개를 각각 model, scaler, features 변수에 저장

            # ── Step 4: 2026년 1월 종목 선정 ─────────────────
            print("\n" + "=" * 60)
            print("Step 4. 2026년 1월 투자 종목 선정")
            print("=" * 60)
            portfolio_jan = select_stocks_jan2026_v2(model, scaler, features)

            if not portfolio_jan.empty:
                # ── Step 5: 2월 성과 확인 ─────────────────────
                print("\n" + "=" * 60)
                print("Step 5. 2026년 2월 성과 확인")
                print("=" * 60)
                feb_results = check_feb_results_v2(portfolio_jan)

                # ── Step 6: 시각화 ─────────────────────────────
                print("\n" + "=" * 60)
                print("Step 6. 결과 시각화")
                print("=" * 60)
                try:
                    visualize_results(feb_results, df_train, portfolio_jan)
                except Exception as e:
                    print(f"⚠️ 시각화 실패: {e}")

                # ── 최종 결과 요약 출력 ────────────────────────
                print("\n" + "🎉"*30)
                print("🎉 전체 파이프라인 완료!")
                print(f"📊 학습 데이터: {len(df_train)}개 샘플 (3개년)")
                print(f"🤖 모델: RF + GBM + LightGBM 앙상블")
                print(f"💰 평균 수익률: {feb_results['수익률(%)'].mean():.2f}%")
                print(f"✅ 승률: {len(feb_results[feb_results['수익률(%)']>0])/len(feb_results)*100:.1f}%")
                print("🎉"*30)
            else:
                print("❌ 포트폴리오 선정 실패")
