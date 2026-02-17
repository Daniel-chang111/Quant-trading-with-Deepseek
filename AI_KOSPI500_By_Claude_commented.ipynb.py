# ============================================================
# 🚀 코스피 종목 데이터 수집 + AI 학습 시스템 (버그 수정판)
# ============================================================
# 📌 이 프로그램이 하는 일:
#   1. 코스피 500개 종목의 2025년 1~12월 주식 데이터 수집
#   2. 수집한 데이터로 AI(랜덤포레스트) 모델 학습
#      → "이런 특징의 종목이 다음달에 오른다"는 패턴을 배움
#   3. 2026년 1월 기준 상승 가능성 TOP 20 종목 선정
#   4. 2026년 2월 실제 성과 확인
# ============================================================


# ============================================================
# 0. 필요한 도구(패키지) 설치 및 불러오기
# ============================================================

# !pip install : 외부 도구를 인터넷에서 설치하는 명령어
# -q : 설치 과정 메시지를 간략하게 출력 (quiet 모드)
!pip install pykrx pandas numpy scikit-learn -q

import numpy as np      # 숫자 계산 전문 도구 (평균, 표준편차 등)
import pandas as pd     # 표(데이터프레임) 처리 도구 (엑셀과 비슷한 역할)
from datetime import datetime, timedelta
# datetime  : 날짜를 다루는 도구 (예: '20250131' 문자열 → 날짜 객체로 변환)
# timedelta : 날짜 계산 도구 (예: 오늘 날짜 - 200일 = 200일 전 날짜)

from pykrx import stock
# pykrx : 한국 주식 데이터(KRX)를 무료로 가져오는 파이썬 라이브러리
# stock : pykrx 안의 주식 관련 함수 묶음

import time      # 시간 측정, 일정 시간 멈추기(sleep) 등에 사용
import os        # 폴더 만들기, 파일 존재 여부 확인 등 파일 관련 작업
import pickle    # 파이썬 데이터를 파일로 저장하고 다시 불러오는 도구
import warnings  # 경고 메시지 제어 도구

from concurrent.futures import ThreadPoolExecutor, as_completed
# ThreadPoolExecutor : 여러 작업을 동시에 실행하는 도구 (병렬 처리)
#   비유: 500개 종목을 한 명이 순서대로 조회하는 대신
#         3명이 나눠서 동시에 조회 → 3배 빠름
# as_completed : 병렬 작업이 끝나는 순서대로 결과를 받아오는 도구

from sklearn.ensemble import RandomForestClassifier
# RandomForestClassifier : 랜덤포레스트 AI 모델
# 200개의 결정나무가 각자 예측하고 다수결로 최종 결정
# UC버클리 Leo Breiman 교수가 2001년에 발표한 알고리즘

from sklearn.model_selection import train_test_split
# train_test_split : 전체 데이터를 학습용/검증용으로 나누는 도구
# 예) 3,486개 데이터 → 학습 2,789개(80%) + 검증 697개(20%)

from sklearn.metrics import accuracy_score, classification_report
# accuracy_score      : 정확도 계산 (맞춘 비율)
# classification_report : 상승/하락 예측 상세 성능 보고서

from sklearn.preprocessing import StandardScaler
# StandardScaler : 특성값들의 범위를 통일하는 정규화 도구
# 예) PER은 1~100, 모멘텀은 -50~+100 → 모두 비슷한 범위로 통일
#     범위가 다르면 AI가 큰 숫자에 편향되어 학습하는 문제 발생

warnings.filterwarnings('ignore')  # 분석과 무관한 경고 메시지 숨기기


# ============================================================
# 📦 캐시(Cache) 시스템
# ============================================================
# 캐시란? 한 번 인터넷에서 받은 데이터를 파일로 저장해두는 것
# 왜 필요? 500개 종목 × 12개월 데이터를 매번 새로 받으면 시간이 매우 오래 걸림
#          캐시에 저장해두면 재실행 시 파일에서 바로 읽어서 훨씬 빠름
# 비유: 도서관에서 책을 빌려 집에 두면 다음에는 도서관을 안 가도 됨

class StockDataCache:

    def __init__(self, cache_dir='stock_cache'):
        # __init__ : 클래스를 처음 만들 때 자동으로 실행되는 초기화 함수
        # cache_dir : 캐시 파일을 저장할 폴더 이름 (기본값: 'stock_cache')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        # os.makedirs : 폴더 생성
        # exist_ok=True : 폴더가 이미 존재해도 오류 없이 넘어감

    def get_cache_path(self, ticker, date):
        # 캐시 파일의 저장 경로를 만드는 함수
        # 예) ticker='005930', date='20250131'
        #     → 'stock_cache/005930_20250131.pkl'
        return f"{self.cache_dir}/{ticker}_{date}.pkl"

    def save(self, ticker, date, data):
        # 데이터를 캐시 파일로 저장하는 함수
        # 'wb' : write binary (이진 형식으로 파일 쓰기)
        with open(self.get_cache_path(ticker, date), 'wb') as f:
            pickle.dump(data, f)    # pickle.dump : 파이썬 데이터 → 파일로 저장

    def load(self, ticker, date):
        # 저장된 캐시 파일을 불러오는 함수
        # 캐시 파일이 없으면 None(없음)을 반환
        path = self.get_cache_path(ticker, date)
        if os.path.exists(path):            # 파일이 존재하면
            with open(path, 'rb') as f:     # 'rb' : read binary (이진 파일 읽기)
                return pickle.load(f)       # 파일 → 파이썬 데이터로 복원
        return None                         # 파일 없으면 None 반환

# 캐시 객체 생성 (이후 코드에서 cache.save(), cache.load()로 사용)
cache = StockDataCache()


# ============================================================
# 📅 유효 거래일 탐색 함수
# ============================================================
# 왜 필요? 주식시장은 주말·공휴일에 쉬어서 해당 날짜 데이터가 없음
# 예) '20250131'이 일요일이면 → '20250130'(금요일) 데이터를 대신 사용
# 최대 5일 전까지 거슬러 올라가며 실제 거래일을 찾아줌

def find_valid_date(target_date, max_days_back=5):
    """
    입력한 날짜에 주식 데이터가 없으면 최대 5일 전까지 탐색해서
    실제 거래일을 찾아주는 함수

    target_date   : 원하는 날짜 (예: '20250131')
    max_days_back : 최대 몇 일 전까지 탐색할지 (기본값: 5일)
    반환값        : 실제 거래일 문자열 또는 None
    """
    # 문자열 '20250131' → 파이썬 날짜 객체로 변환 (날짜 계산을 위해)
    date_obj = datetime.strptime(target_date, '%Y%m%d')
    # '%Y%m%d' : 연도4자리+월2자리+일2자리 형식

    for i in range(max_days_back + 1):  # i = 0, 1, 2, 3, 4, 5 순서로 시도
        check_date = (date_obj - timedelta(days=i)).strftime('%Y%m%d')
        # timedelta(days=i) : i일을 빼서 이전 날짜 계산
        # .strftime('%Y%m%d') : 날짜 객체 → '20250131' 형식 문자열로 변환
        try:
            tickers = stock.get_market_ticker_list(check_date, market="KOSPI")
            # 해당 날짜의 코스피 전체 종목 리스트 요청
            # 거래일이면 종목 목록이 오고, 휴장일이면 빈 리스트가 옴
            if tickers and len(tickers) > 0:   # 종목이 1개라도 있으면 거래일!
                if i > 0:  # 날짜가 보정된 경우에만 안내 메시지 출력
                    print(f"  📅 {target_date} → {check_date}로 보정 ({len(tickers)}개 종목)")
                return check_date  # 유효한 거래일을 찾으면 반환하고 종료
        except:
            continue    # 오류가 나도 다음 날짜로 계속 시도
    return None         # 5일 다 시도해도 거래일을 못 찾으면 None 반환


# ============================================================
# 🔍 단일 종목 데이터 수집 함수
# ============================================================
# 이 함수가 하는 일:
#   종목 하나에 대해 5가지 특성값(PER, PBR, 모멘텀3종, 변동성)을 계산
#   이 특성값들이 AI 모델이 학습하는 '재료'가 됨
#
# 특성값 설명:
#   PER(주가수익비율) : 주가 / 주당순이익 → 낮을수록 이익 대비 저평가
#   PBR(주가순자산비율): 주가 / 주당순자산 → 1 미만이면 자산보다 싸게 거래
#   모멘텀_1m : 최근 1개월 주가 변화율(%) → 단기 추세
#   모멘텀_3m : 최근 3개월 주가 변화율(%) → 중기 추세
#   모멘텀_6m : 최근 6개월 주가 변화율(%) → 장기 추세
#   변동성    : 주가가 얼마나 들쭉날쭉한지 → 클수록 위험

def fetch_ticker_data(ticker, date):
    """
    특정 종목의 특성 데이터를 수집하는 함수

    ticker : 종목코드 (예: '005930' = 삼성전자)
    date   : 기준 날짜 (예: '20250131')
    반환값 : 특성값 딕셔너리 또는 None(수집 실패 시)
    """

    # 캐시에 이미 저장된 데이터가 있는지 먼저 확인
    cached = cache.load(ticker, date)
    if cached is not None:  # None이 아니면 (캐시가 있으면)
        return cached        # API 호출 없이 바로 캐시 반환
    # ⚠️ 주의: `if cached`가 아닌 `if cached is not None`을 써야 함
    #           `if cached`는 빈 딕셔너리 {}도 False로 처리해서 버그 발생

    try:
        # 종목코드 → 종목명 변환 (예: '005930' → '삼성전자')
        name = stock.get_market_ticker_name(ticker)

        # ── 재무 지표 수집 (PER, PBR) ────────────────────────
        df_fund = stock.get_market_fundamental(date, date, ticker)
        # get_market_fundamental : 특정 날짜의 재무 지표 가져오기
        # 날짜를 두 번 쓰는 이유 : (시작일, 종료일) 형식이라 같은 날 조회 시 동일하게 입력

        if df_fund is None or df_fund.empty:
            return None     # 재무 데이터가 없으면 이 종목은 건너뜀

        fund_row = df_fund.iloc[0]  # 첫 번째 행(1일치 데이터이므로 딱 1행)

        # 컬럼 이름 목록 먼저 확인 (pykrx 버전에 따라 컬럼명이 다를 수 있음)
        cols = fund_row.index.tolist()
        per = fund_row['PER'] if 'PER' in cols else np.nan
        pbr = fund_row['PBR'] if 'PBR' in cols else np.nan
        # np.nan : 숫자가 아님(Not a Number) = 데이터 없음을 표현하는 특수값
        # pd.isna : 값이 nan인지 확인하는 함수

        # PER 또는 PBR이 없거나 0 이하이면 분석 불가 → 건너뜀
        if pd.isna(per) or pd.isna(pbr) or per <= 0 or pbr <= 0:
            return None

        # ── 가격 데이터 수집 (약 200 거래일치) ───────────────
        start_date = (datetime.strptime(date, '%Y%m%d') - timedelta(days=200)).strftime('%Y%m%d')
        # 200일 전 날짜를 계산 (영업일 기준 약 140일이지만 여유 있게 200일로 설정)
        df_price = stock.get_market_ohlcv_by_date(start_date, date, ticker)
        # get_market_ohlcv_by_date : 기간별 가격/거래량 데이터 반환
        # OHLCV = Open(시가), High(고가), Low(저가), Close(종가), Volume(거래량)

        if df_price is None or len(df_price) < 100:
            return None     # 100 거래일치 미만이면 지표 계산 불가 → 건너뜀

        closes = df_price['종가']   # 매일의 종가(마감 가격) 시리즈만 따로 저장

        # ── 모멘텀 계산 (주가 추세) ───────────────────────────
        # .iloc[-1]  : 가장 최근(마지막) 값
        # .iloc[-22] : 22번째 전 값 = 영업일 기준 약 1개월 전
        # .iloc[-66] : 66번째 전 값 = 영업일 기준 약 3개월 전
        momentum_1m = (closes.iloc[-1] / closes.iloc[-22]  - 1) * 100  # 1개월 수익률(%)
        momentum_3m = (closes.iloc[-1] / closes.iloc[-66]  - 1) * 100  # 3개월 수익률(%)
        momentum_6m = (closes.iloc[-1] / closes.iloc[-132] - 1) * 100 if len(df_price) > 132 else 0
        # 6개월(132일치) 데이터가 없으면 0으로 처리

        # ── 변동성 계산 ───────────────────────────────────────
        # 변동성 = 주가가 얼마나 불규칙하게 오르내리는지 나타내는 수치
        # 최근 60 거래일의 '등락률' 표준편차로 계산
        # .tail(60) : 마지막 60개 데이터만 선택
        # .std()    : 표준편차 계산 (값이 클수록 변동이 심함 = 위험한 주식)
        volatility = df_price['등락률'].tail(60).std()

        # ── 결과를 딕셔너리로 정리 ────────────────────────────
        # 딕셔너리 = {키: 값} 형태의 데이터 구조
        result = {
            '티커':      ticker,                        # 종목코드 (예: '005930')
            '종목':      name,                          # 종목명  (예: '삼성전자')
            'PER':       round(per, 2),                 # 소수점 2자리로 반올림
            'PBR':       round(pbr, 2),
            '모멘텀_1m': round(momentum_1m, 2),         # 1개월 수익률(%)
            '모멘텀_3m': round(momentum_3m, 2),         # 3개월 수익률(%)
            '모멘텀_6m': round(momentum_6m, 2),         # 6개월 수익률(%)
            '변동성':    round(volatility, 2),
            '현재가':    closes.iloc[-1]                # 가장 최근 종가
        }

        cache.save(ticker, date, result)    # 수집한 데이터를 캐시 파일로 저장
        return result

    except Exception as e:
        return None     # 어떤 오류가 발생해도 None 반환 (이 종목은 건너뜀)


# ============================================================
# 📅 월별 데이터 수집 함수 (병렬 처리)
# ============================================================
# max_workers=3 : 동시에 3개 종목씩 병렬 수집
# 왜 10이 아닌 3? : 너무 많은 동시 요청은 KRX 서버에서 차단할 수 있음
# 50개마다 1초 쉬기 : 서버 과부하 방지 (Rate Limit 보호)

def collect_month_data(date, max_workers=3):
    """
    특정 월의 코스피 상위 500개 종목 데이터를 병렬로 수집하는 함수

    date        : 수집 기준 날짜 (월말 거래일, 예: '20250131')
    max_workers : 동시에 처리할 종목 수 (기본값: 3)
    반환값      : 수집된 종목 데이터가 담긴 데이터프레임(표)
    """
    print(f"📅 {date} 수집 시작...")

    try:
        all_tickers = stock.get_market_ticker_list(date, market="KOSPI")
        # 해당 날짜의 코스피 전체 종목 코드 리스트 가져오기
        if not all_tickers or len(all_tickers) == 0:
            print(f"  ⚠️ {date}: 종목 없음 (휴장일 가능성)")
            return pd.DataFrame()   # 빈 표 반환
        tickers = all_tickers[:500]  # 앞에서 500개만 사용
        print(f"  📋 수집 대상: {len(tickers)}개")
    except Exception as e:
        print(f"  ❌ 종목 리스트 조회 실패: {e}")
        return pd.DataFrame()

    results = []        # 수집 결과를 담을 빈 리스트
    success_count = 0   # 성공한 종목 수 카운터

    # 병렬 처리 시작
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.submit : 각 종목 수집 작업을 병렬 대기열에 등록
        futures = {executor.submit(fetch_ticker_data, ticker, date): ticker
                   for ticker in tickers}
        # futures : {작업객체: 종목코드} 형태의 딕셔너리

        for i, future in enumerate(as_completed(futures)):
            # as_completed : 먼저 완료된 작업부터 순서대로 결과 수령
            # enumerate : 순번(i)과 작업(future)을 함께 제공
            result = future.result()    # 완료된 작업의 결과값 가져오기
            if result:                  # 결과가 있으면 (None이 아니면)
                result['기준일'] = date # 어느 달 데이터인지 날짜 컬럼 추가
                results.append(result)  # 결과 리스트에 추가
                success_count += 1      # 성공 카운터 증가

            # 50개 처리마다 1초 쉬기 (KRX 서버 과부하 방지)
            if i % 50 == 49:
                time.sleep(1)

    df = pd.DataFrame(results)  # 리스트 → 데이터프레임(표)으로 변환
    print(f"  ✅ 성공: {success_count}개 / 전체: {len(tickers)}개")
    return df


# ============================================================
# 📊 2025년 전체 데이터 수집 함수 (1월~12월)
# ============================================================
# 월말 기준일을 사용하는 이유:
#   월말 데이터로 "이번달 말 상태"를 기록하고
#   "다음달 말 가격"과 비교해서 수익률을 계산하기 위함
# find_valid_date()를 여기서 적용하는 것이 핵심 버그 수정 포인트!

def collect_2025_full_data():
    """
    2025년 1월~12월 월말 기준으로 코스피 500개 종목 데이터를 수집하는 함수
    반환값 : 12개월 × 약 300~326개 종목 = 약 3,800개 행의 데이터프레임
    """

    # 2025년 각 월말 날짜 목록 (실제 거래일이 아닐 수 있어 아래에서 보정)
    month_targets = [
        '20250131', '20250228', '20250331', '20250430',
        '20250530', '20250630', '20250731', '20250829',
        '20250930', '20251031', '20251128', '20251230'
    ]

    print("📊 2025년 실제 거래일 확인중...")
    valid_months = []   # 실제 거래일로 보정된 날짜를 담을 리스트

    for m in month_targets:
        valid = find_valid_date(m)  # 실제 거래일 탐색 (핵심 버그 수정!)
        # 수정 전: month_targets를 그대로 사용 → 휴장일이면 데이터 없음
        # 수정 후: find_valid_date()로 가장 가까운 실제 거래일로 자동 보정
        if valid:
            valid_months.append(valid)
        else:
            print(f"  ❌ {m}: 유효한 거래일 없음")

    print(f"\n✅ 수집할 월: {valid_months}\n")

    all_data = []   # 각 월의 데이터프레임을 담을 리스트
    start_time = time.time()    # 시작 시간 기록 (소요 시간 계산용)

    for i, month in enumerate(valid_months, 1):  # 1부터 시작하는 순번(i)
        print(f"\n[{i}/{len(valid_months)}] ", end="")
        df_month = collect_month_data(month)    # 해당 월 데이터 수집

        if df_month is not None and len(df_month) > 0:
            all_data.append(df_month)   # 수집 성공 시 리스트에 추가

        # 월별 수집 사이에 2초 대기 (KRX 서버 부하 분산)
        time.sleep(2)

        # 남은 시간 예측 계산
        elapsed   = time.time() - start_time          # 지금까지 걸린 시간(초)
        remaining = (elapsed / i) * (len(valid_months) - i)  # 예상 남은 시간
        print(f"  ⏱️ 경과: {elapsed:.0f}초 / 잔여: {remaining:.0f}초")

    if all_data:
        # 12개월 데이터를 하나의 큰 표로 합치기
        final_df = pd.concat(all_data, ignore_index=True)
        # pd.concat     : 여러 데이터프레임을 세로로 이어 붙이기
        # ignore_index  : 인덱스를 0, 1, 2, ...으로 새로 부여
        print(f"\n✅ 수집 완료: {len(final_df)}개 샘플, {final_df['기준일'].nunique()}개월")
        # .nunique() : 고유값 개수 (중복 제외한 월 수)
        return final_df
    else:
        print("❌ 수집 실패: 모든 월에서 데이터 없음")
        print("💡 pykrx 버전 확인: pip show pykrx")
        print("💡 직접 테스트: stock.get_market_ticker_list('20250131', market='KOSPI')")
        return pd.DataFrame()   # 빈 데이터프레임 반환


# ============================================================
# 🏷️ 다음달 수익률(레이블) 추가 함수
# ============================================================
# AI 모델은 "문제(특성값)"와 "정답(레이블)"을 같이 줘야 학습할 수 있음
# 여기서 정답 = 다음달에 주가가 올랐는지(1) 내렸는지(0)
#
# 예시:
#   1월 말 삼성전자 가격: 70,000원
#   2월 말 삼성전자 가격: 73,500원
#   다음달수익률 = (73,500 / 70,000 - 1) × 100 = +5%  → target = 1 (상승)
#
#   1월 말 LG전자 가격: 100,000원
#   2월 말 LG전자 가격:  95,000원
#   다음달수익률 = (95,000 / 100,000 - 1) × 100 = -5% → target = 0 (하락)

def add_future_returns_full(df):
    """
    각 종목의 다음달 수익률을 계산하고 상승/하락 레이블을 추가하는 함수

    df      : 월별 종목 데이터프레임 (collect_2025_full_data() 반환값)
    반환값  : 다음달수익률과 target(0 또는 1) 컬럼이 추가된 데이터프레임
    """
    if df.empty:
        return pd.DataFrame()

    results = []

    # 종목별로 묶어서 처리 (같은 종목의 월별 데이터를 시간 순으로 비교)
    for ticker, group in df.groupby('티커'):
        # df.groupby('티커') : 종목코드가 같은 행들을 묶음
        group = group.sort_values('기준일').reset_index(drop=True)
        # sort_values('기준일') : 날짜 오름차순 정렬 (1월 → 2월 → ... → 12월)
        # reset_index(drop=True) : 인덱스를 0, 1, 2, ...으로 재설정

        if len(group) < 2:
            continue    # 2개월치 미만이면 다음달 수익률 계산 불가 → 건너뜀

        for i in range(len(group) - 1):  # 마지막 월은 제외 (다음달 데이터 없음)
            row        = group.iloc[i].to_dict()      # i번째 월 데이터 → 딕셔너리
            next_price = group.iloc[i + 1]['현재가']  # i+1번째 월(다음달) 가격

            # 수익률 계산
            ret = (next_price / row['현재가'] - 1) * 100
            row['다음달수익률'] = round(ret, 2)
            row['target'] = 1 if ret > 0 else 0  # 양수면 1(상승), 음수면 0(하락)
            results.append(row)

    df_result = pd.DataFrame(results)
    print(f"✅ 학습 샘플: {len(df_result)}개 / 종목 수: {df_result['티커'].nunique()}개")
    return df_result


# ============================================================
# 🤖 AI 모델 학습 함수
# ============================================================
# 여기서 하는 일:
#   1. 5가지 특성(PER, PBR, 모멘텀3종, 변동성)을 AI의 입력으로 준비
#   2. 정규화로 특성값의 범위를 통일
#   3. 80% 학습 / 20% 검증으로 데이터 분할
#   4. 랜덤포레스트 모델 학습 (200그루 나무)
#   5. 검증 데이터로 성능 평가

def train_ai_model_full(df_train):
    """
    AI 모델을 학습하고 성능을 평가하는 함수

    df_train : 레이블(target)이 포함된 학습 데이터프레임
    반환값   : (학습된 모델, 정규화 기준, 특성 목록) 튜플
    """

    # AI가 참고할 5가지 특성 목록
    feature_cols = ['PER', 'PBR', '모멘텀_1m', '모멘텀_3m', '변동성']

    # 특성값 중 하나라도 빠진(NaN) 행 제거 (불완전한 데이터는 학습에 방해)
    df_train = df_train.dropna(subset=feature_cols)

    # 특성 행렬(X)과 정답 벡터(y) 분리
    X = df_train[feature_cols].values   # 2D 배열: 행=샘플, 열=특성
    y = df_train['target'].values       # 1D 배열: 0 또는 1

    # 정규화: 각 특성의 범위를 평균 0, 표준편차 1로 통일
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # fit_transform : 정규화 기준을 학습 데이터로 계산하고 동시에 변환

    # 학습/검증 분할 (80% 학습, 20% 검증)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y,
        test_size=0.2,      # 20%를 검증용으로
        random_state=42,    # 난수 시드 고정 (실행마다 같은 분할 결과)
        stratify=y          # 상승/하락 비율을 학습/검증 셋에서 동일하게 유지
    )

    # 랜덤포레스트 모델 설정
    model = RandomForestClassifier(
        n_estimators=200,       # 결정 나무 200그루 사용
        max_depth=10,           # 나무 최대 깊이 10단계
        min_samples_split=5,    # 가지를 나누려면 최소 5개 샘플 필요
        random_state=42,        # 결과 재현을 위한 난수 시드
        n_jobs=-1               # 모든 CPU 코어 사용해서 빠르게 학습
    )

    # 모델 학습 (패턴 학습 단계)
    model.fit(X_train, y_train)
    # 학습 데이터(X_train)와 정답(y_train)을 보여주며 패턴을 학습

    # 검증 데이터로 성능 평가
    y_pred = model.predict(X_val)   # 검증 데이터로 상승/하락 예측
    acc    = accuracy_score(y_val, y_pred)  # 정확도 계산
    print(f"✅ 검증 정확도: {acc:.2%}")    # .2% : 소수를 퍼센트로 변환

    # 상세 성능 보고서 출력
    print(classification_report(y_val, y_pred, target_names=['하락', '상승']))
    # precision : 상승이라고 예측한 것 중 실제 상승 비율 (정밀도)
    # recall    : 실제 상승 중 상승이라고 맞춘 비율 (재현율)
    # f1-score  : precision과 recall의 조화 평균

    return model, scaler, feature_cols
    # 세 값을 동시에 반환 → 호출하는 쪽에서 model, scaler, features = ... 로 받음


# ============================================================
# 💰 2026년 1월 투자 종목 선정 함수
# ============================================================
# 학습된 AI 모델로 2026년 1월 기준 각 종목의 상승 확률을 계산하고
# 상위 20개 종목을 선정

def select_stocks_jan2026(model, scaler, features, date='20260130'):
    """
    AI 모델로 2026년 1월 기준 상승 가능성 상위 20개 종목을 선정하는 함수

    model    : 학습된 랜덤포레스트 모델
    scaler   : 학습 시 사용한 정규화 기준 (반드시 동일한 기준 적용!)
    features : 사용할 특성 목록
    date     : 기준 날짜 (기본값: 2026년 1월 30일)
    반환값   : 상위 20개 종목 데이터프레임
    """

    # 실제 거래일 탐색
    valid_date = find_valid_date(date)
    if not valid_date:
        print("❌ 유효한 거래일 없음")
        return pd.DataFrame()

    # 코스피 종목 리스트 가져오기 (앞에서 500개)
    tickers = stock.get_market_ticker_list(valid_date, market="KOSPI")[:500]
    stock_data = []

    # 각 종목 데이터 수집 (순차적으로)
    for ticker in tickers:
        try:
            name     = stock.get_market_ticker_name(ticker)
            df_fund  = stock.get_market_fundamental(valid_date, valid_date, ticker)
            if df_fund is None or df_fund.empty:
                continue

            fund_row = df_fund.iloc[0]
            per = fund_row.get('PER', np.nan)   # .get() : 키가 없으면 기본값 반환
            pbr = fund_row.get('PBR', np.nan)

            if pd.isna(per) or pd.isna(pbr) or per <= 0:
                continue

            start_date = (datetime.strptime(valid_date, '%Y%m%d') - timedelta(days=200)).strftime('%Y%m%d')
            df_price   = stock.get_market_ohlcv_by_date(start_date, valid_date, ticker)
            if df_price is None or len(df_price) < 100:
                continue

            closes = df_price['종가']

            # 종목 데이터를 딕셔너리로 정리
            stock_data.append({
                '종목':      name,
                '티커':      ticker,
                'PER':       per,
                'PBR':       pbr,
                '모멘텀_1m': (closes.iloc[-1] / closes.iloc[-22] - 1) * 100,
                '모멘텀_3m': (closes.iloc[-1] / closes.iloc[-66] - 1) * 100,
                '변동성':    df_price['등락률'].tail(60).std(),
                '1월말_가격': closes.iloc[-1]   # 2월 수익률 계산에 사용할 매수 기준가
            })
        except:
            continue    # 오류가 나는 종목은 건너뜀

    df_jan = pd.DataFrame(stock_data)
    if df_jan.empty:
        return pd.DataFrame()

    # AI 점수 계산
    X_jan = scaler.transform(df_jan[features].values)
    # 학습 때와 동일한 정규화 기준(scaler) 적용 (중요!)
    # 다른 기준으로 정규화하면 AI가 엉뚱한 예측을 함

    df_jan['AI_점수'] = model.predict_proba(X_jan)[:, 1]
    # predict_proba : 각 종목의 [하락확률, 상승확률] 계산
    # [:, 1]        : 모든 행에서 1번 열(상승확률)만 추출 → AI_점수
    # AI_점수 0.7 = "이 종목은 다음달에 70% 확률로 오를 것"으로 AI가 판단

    # AI 점수 기준 상위 20개 선정
    top_20 = df_jan.nlargest(20, 'AI_점수')
    # nlargest(20, 'AI_점수') : AI_점수가 높은 순서로 20개 선택

    print(top_20[['종목', 'PER', 'PBR', '모멘텀_1m', 'AI_점수']].to_string(index=False))
    return top_20


# ============================================================
# 📈 2026년 2월 성과 확인 함수
# ============================================================
# 1월에 선정한 20개 종목을 1월 말 가격에 샀다고 가정하고
# 2월 특정일의 가격과 비교해서 실제 수익률 계산

def check_feb_results(portfolio_jan, date='20260213'):
    """
    1월 선정 종목들의 2월 실제 수익률을 확인하는 함수

    portfolio_jan : 1월에 선정된 TOP 20 포트폴리오 데이터프레임
    date          : 성과 확인 날짜 (기본값: 2026년 2월 13일)
    반환값        : 종목별 수익률이 담긴 데이터프레임
    """
    results = []

    for _, row in portfolio_jan.iterrows():
        # iterrows() : 데이터프레임을 한 행씩 순서대로 읽기
        # _ : 인덱스 번호 (사용 안 하므로 _로 표시)
        try:
            # 2월 특정일의 종가 데이터 가져오기
            df_feb    = stock.get_market_ohlcv_by_date(date, date, row['티커'])
            if df_feb.empty:
                continue

            # 수익률 = (2월 종가 / 1월 말 매수가 - 1) × 100
            ret = (df_feb['종가'].iloc[0] / row['1월말_가격'] - 1) * 100
            results.append({
                '종목':      row['종목'],
                '수익률(%)': round(ret, 2)
            })
        except:
            continue

    # 수익률 기준 내림차순 정렬 (높은 수익률이 위로)
    df_res = pd.DataFrame(results).sort_values('수익률(%)', ascending=False)
    # ascending=False : 내림차순 (큰 값 → 작은 값)

    # 결과 출력
    print(df_res.to_string(index=False))    # index=False : 행 번호 숨기기
    print(f"\n평균 수익률: {df_res['수익률(%)'].mean():.2f}%")
    print(f"승률: {len(df_res[df_res['수익률(%)']>0])/len(df_res)*100:.1f}%")
    # 승률 = 수익률이 0% 초과인 종목 수 / 전체 종목 수 × 100
    return df_res


# ============================================================
# 🚀 메인 실행 블록 (프로그램 시작점)
# ============================================================
# 아래가 실제로 실행되는 순서:
#   Step 1. 2025년 전체 데이터 수집
#   Step 2. 다음달 수익률 레이블 생성
#   Step 3. AI 모델 학습
#   Step 4. 2026년 1월 투자 종목 선정
#   Step 5. 2026년 2월 성과 확인

# Step 1. 2025년 전체 데이터 수집 (약 10~15분 소요)
df_2025_full = collect_2025_full_data()

# 수집 성공 여부 확인 후 다음 단계 진행
if not df_2025_full.empty:  # 데이터가 있으면 (비어 있지 않으면)

    # Step 2. 다음달 수익률 레이블 생성
    df_train = add_future_returns_full(df_2025_full)

    if not df_train.empty:

        # Step 3. AI 모델 학습
        model, scaler, features = train_ai_model_full(df_train)
        # 반환값 3개를 각각 model, scaler, features 변수에 저장

        # Step 4. 2026년 1월 투자 종목 선정
        portfolio_jan = select_stocks_jan2026(model, scaler, features)

        if not portfolio_jan.empty:

            # Step 5. 2026년 2월 성과 확인
            feb_results = check_feb_results(portfolio_jan)
            # 최종 결과: 평균 수익률, 승률 출력
