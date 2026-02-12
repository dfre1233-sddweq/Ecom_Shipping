# 🚚 배송의 민족 — E-Commerce Shipping Analytics

> **배송 지연 예측 프로젝트** | E-Commerce 고객 주문의 배송 지연 여부를 예측하는 이진 분류 모델링 프로젝트

---

## 📋 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **팀명** | 배송의 민족 |
| **팀원** | [박찬영](https://github.com/dfre1233-sddweq), [이경욱](https://github.com/eduGWL), [지소연](https://github.com/happysoyeon09-design), [신우철](https://github.com/chul3224) |
| **발표일** | 2026년 02월 13일 |
| **데이터셋** | [E-Commerce Shipping Dataset (Kaggle)](https://www.kaggle.com/datasets/prachi13/customer-analytics/data) |
| **목표** | 고객 주문의 **배송 지연 여부** 예측 (이진 분류) |
| **최종 노트북** | `customer_analytics_modeling_ver_3_0.ipynb` |

---

## 🎯 프로젝트 목표

E-Commerce 환경에서 배송 지연은 고객 만족도와 재구매율에 직접적인 영향을 미칩니다. 본 프로젝트는 주어진 주문 및 배송 관련 데이터를 분석하여 **배송 지연 가능성을 사전에 예측**하는 머신러닝 모델을 구축합니다.

- 다양한 머신러닝 알고리즘을 비교·평가하여 최적의 예측 모델을 선정
- **SHAP** 분석을 통해 모델의 예측 근거를 해석
- 배송 지연에 영향을 미치는 **핵심 요인**을 도출

---

## 📊 데이터셋

- **출처**: [Kaggle - E-Commerce Shipping Dataset](https://www.kaggle.com/datasets/prachi13/customer-analytics/data)
- **샘플 수**: 10,999건
- **결측치**: 없음
- **타겟 변수**: `Reached.on.Time_Y.N` (1: 배송 지연, 0: 정시 도착)

### 컬럼 설명

| 컬럼명 | 설명 | 타입 |
|--------|------|------|
| `ID` | 고유 식별자 | int |
| `Warehouse_block` | 창고 블록 (A~F) | object |
| `Mode_of_Shipment` | 배송 수단 (Ship, Flight, Road) | object |
| `Customer_care_calls` | 고객 문의 횟수 | int |
| `Customer_rating` | 고객 평점 (1~5) | int |
| `Cost_of_the_Product` | 상품 가격 | int |
| `Prior_purchases` | 이전 구매 횟수 | int |
| `Product_importance` | 상품 중요도 (Low, Medium, High) | object |
| `Gender` | 성별 (M, F) | object |
| `Discount_offered` | 할인율 (%) | int |
| `Weight_in_gms` | 상품 무게 (g) | int |
| `Reached.on.Time_Y.N` | 배송 지연 여부 (1=지연, 0=정시) | int |

---

## 🔬 분석 파이프라인

```
📥 데이터 로딩
    ↓
📊 탐색적 데이터 분석 (EDA) — 타겟 분포, 수치형/범주형 시각화, 상관관계, 다변량 교차 분석
    ↓
🔧 전처리 — 타겟 boolean 변환, 원핫인코딩, 다운캐스팅
    ↓
⚙️ 특성 공학 — 9개 파생변수 생성 (할인/무게 구간, 상호작용, 비율 등)
    ↓
📐 데이터 분할 & 스케일링 — Train/Test 분할, StandardScaler 적용
    ↓
🤖 베이스라인 모델 비교 — 8개 알고리즘 성능 비교
    ↓
🎯 Optuna 하이퍼파라미터 튜닝 — 상위 3개 모델 최적화
    ↓
📈 모델 평가 — Confusion Matrix, ROC, PR Curve, 특성 중요도, SHAP
    ↓
💾 모델 저장 — best_model.pkl, scaler.pkl
```

### 분석 상세 목차

| No. | 섹션 | 설명 |
|-----|------|------|
| 0 | 데이터셋 소개 | 컬럼 설명 및 분석 파이프라인 |
| 1 | 라이브러리 임포트 & 데이터 로딩 | 환경 설정 및 데이터 확인 |
| 2 | 데이터 품질 체크 | 결측치 확인, 중복값 확인, 이상치 탐지 |
| 3 | 탐색적 데이터 분석 (EDA) | 타겟 분포, 수치형/범주형 시각화, 상관관계, 다변량 교차 분석 |
| 4 | 전처리 | 타겟 boolean 변환, 원핫인코딩, 다운캐스팅 |
| 5 | 특성 공학 | 파생변수 생성 및 타겟 상관관계 분석 |
| - | 데이터 분할 & 스케일링 | Train/Test 분할, StandardScaler 적용 |
| 6 | 베이스라인 모델 비교 | 8개 알고리즘 성능 비교 |
| 7 | Optuna 하이퍼파라미터 튜닝 | 상위 3개 모델 최적화 |
| 8 | 모델 평가 | Confusion Matrix, ROC, PR Curve, 특성 중요도, SHAP |
| 9 | 전체 결과 요약 | 베이스라인 + 튜닝 통합 비교 |
| 10 | 모델 저장 | best_model.pkl, scaler.pkl |
| 11 | 결론 | 주요 발견사항 · 한계점 · 향후 개선 방향 |

---

## 🔍 주요 발견사항

### 1. 배송 지연의 핵심 요인: 할인율과 상품 무게

EDA부터 SHAP 분석까지 일관되게, **`Discount_offered`(할인율)** 과 **`Weight_in_gms`(상품 무게)** 가 배송 지연을 결정짓는 가장 중요한 변수로 확인되었습니다.

| 발견 | 근거 |
|------|------|
| 할인율 10% 초과 시 배송 지연 급증 | 파생변수 `Is_HighDiscount` ↔ 타겟 상관계수 **+0.463** |
| 초중량(4,500g+) 상품은 오히려 정시 도착 경향 | `Weight_bin_초중량` ↔ 타겟 상관계수 **-0.291** |
| 높은 할인 + 저무게 조합에서 지연 집중 | 인터랙티브 산점도 및 Violin Plot에서 시각적 확인 |

### 2. 고객 속성은 배송 지연과 무관

`Customer_care_calls`, `Customer_rating`, `Prior_purchases`, `Gender` 등 **고객 관련 변수들은 타겟과의 상관관계가 거의 0에 가까웠습니다**. 파생변수(`Loyalty_score`, `Calls_per_Purchase`)를 생성해 보았으나 마찬가지로 예측력이 없었으며, 배송 지연은 **고객 특성이 아닌 주문/물류 특성**에 의해 결정됨을 확인하였습니다.

### 3. 모델 성능

- 8개 베이스라인 모델 비교 결과, **Gradient Boosting**이 F1-Score 기준 최고 성능
- Optuna 하이퍼파라미터 튜닝을 통해 성능 추가 개선
- 전체 모델의 F1-Score는 **0.64~0.68 범위**에 분포
- 데이터 자체의 분류 난이도가 높은 과제임을 확인

---

## 🛠️ 사용 기술 스택

### 언어 & 환경
- **Python 3.11**
- **Jupyter Notebook**

### 데이터 분석 & 시각화
- `numpy`, `pandas` — 데이터 처리
- `matplotlib`, `seaborn` — 정적 시각화
- `plotly` — 인터랙티브 시각화

### 머신러닝 & 모델링
- `scikit-learn` — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN 등
- `XGBoost` — Extreme Gradient Boosting
- `LightGBM` — Light Gradient Boosting Machine
- `CatBoost` — Categorical Boosting

### 하이퍼파라미터 튜닝
- `Optuna` — 베이지안 기반 하이퍼파라미터 최적화 (50 trials × 3 모델)

### 모델 해석
- `SHAP` — SHapley Additive exPlanations

### 모델 저장
- `joblib` — 모델 직렬화

---

## 📁 프로젝트 구조

```
Ecom_Shipping/
│
├── 📄 README.md                                    # 프로젝트 설명 (본 파일)
├── 📓 customer_analytics_modeling_ver_3_0.ipynb     # 최종 분석 노트북
├── 🤖 best_model.pkl                               # 최적 모델 (저장)
├── 📏 scaler.pkl                                   # StandardScaler (저장)
│
├── 📂 data/                                        # 데이터셋 폴더
│   ├── Train.csv                                   # 학습 데이터
│   └── customer-analytics.zip                      # 원본 압축 파일
│
├── 📂 LGU/                                         # 이경욱 개인 작업 폴더
│   ├── ECS_Data.ipynb
│   ├── ECS_EDA.ipynb
│   ├── ECS_FE.ipynb
│   ├── ECS_ML.ipynb
│   ├── Project_Presentation.ipynb
│   └── ...
│
├── 📂 PCY/                                         # 박찬영 개인 작업 폴더
│   ├── Modeling(Final1[No_Customer_Vars]).ipynb
│   ├── Modeling(Final2[Customer_Vars]).ipynb
│   └── ...
│
├── 📂 soyeon ji/                                   # 지소연 개인 작업 폴더
│   ├── Advanced_EDA.ipynb
│   ├── EDA.ipynb
│   ├── Key_Features_evi.ipynb
│   ├── modeling.ipynb
│   ├── preprocessing.ipynb
│   └── Presentation_final.ipynb
│
├── 📂 woochul/                                     # 신우철 개인 작업 폴더
│   ├── ECS_Woochul.ipynb
│   ├── customer_analytics_modeling_final.ipynb
│   ├── eda_Ecom.ipynb
│   └── ...
│
└── 📂 참고파일/                                     # 참고 자료
```

---

## 🚀 실행 방법

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/dfre1233-sddweq/Ecom_Shipping.git
cd Ecom_Shipping

# 필수 라이브러리 설치
pip install numpy pandas matplotlib seaborn plotly scikit-learn xgboost lightgbm catboost optuna shap joblib
```

### 2. 데이터 준비

- `data/` 폴더에 `Train.csv` 파일이 포함되어 있는지 확인
- 혹은 [Kaggle 데이터셋 페이지](https://www.kaggle.com/datasets/prachi13/customer-analytics/data)에서 다운로드

### 3. 노트북 실행

```bash
jupyter notebook customer_analytics_modeling_ver_3_0.ipynb
```

> 📌 **환경 참고**: 노트북은 한글 폰트(`Malgun Gothic`)를 사용하고 있습니다. MacOS에서는 `AppleGothic` 등으로 변경이 필요할 수 있습니다.

---

## 📈 모델 성능 요약

| 구분 | 주요 모델 | F1-Score 범위 |
|------|-----------|--------------|
| 베이스라인 (8개 모델) | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, XGBoost, LightGBM | 0.64 ~ 0.67 |
| Optuna 튜닝 (3개 모델) | 상위 3개 모델 최적화 | 0.66 ~ 0.68 |
| 앙상블 | Voting / Stacking 등 | 0.66 ~ 0.68 |

> **최종 선정 모델**: Gradient Boosting 계열 (Optuna 튜닝 후 F1-Score 기준 최고 성능)

---

## ⚠️ 한계점

### 데이터 한계
- **시계열 정보 부족**: 주문 일자, 배송 일자 등의 시간 정보가 없어 계절성·트렌드를 반영할 수 없음
- **외부 요인 미반영**: 날씨, 교통 상황, 물류센터 처리량 등의 외부 데이터 미포함
- **샘플 크기 제한**: 10,999건의 데이터로는 드문 케이스에 대한 일반화가 어려움

### 모델 한계
- 특성 간 복잡한 비선형 관계를 완전히 포착하기 어려움
- F1-Score 0.68 수준에 머무는 것은 현재 변수만으로는 설명할 수 없는 잠재 요인이 존재함을 시사
- 극단적 할인율, 비정상 무게 등 드문 케이스에 대한 예측 성능 제한적

---

## 🔮 향후 개선 방향

### 데이터 확장
- **시계열 데이터 수집**: 최소 1년 이상의 주문·배송 일자 데이터 확보
- **외부 데이터 통합**: 날씨 API, 교통 정보, 공휴일 데이터 결합
- **고객 피드백 데이터 추가**: 배송 후 만족도, 재주문 여부 등

### 모델 고도화
- **딥러닝 모델 실험**: 시계열 데이터 확보 시 LSTM, Transformer 등 적용
- **앙상블 기법 고도화**: Stacking, Blending 등 다층 앙상블 전략
- **AutoML 도입 검토**: H2O, Auto-sklearn 등을 통한 자동 모델 탐색

### 시스템 구축
- **MLOps 파이프라인 구축**: 모델 학습-배포-모니터링 자동화
- **A/B 테스트 프레임워크**: 모델 적용 효과 정량 검증
- **실시간 예측 API 개발**: 주문 시점에 배송 지연 위험을 사전 예측

---

## 📜 라이선스

본 프로젝트는 학습 목적으로 진행되었으며, 원본 데이터셋은 [Kaggle](https://www.kaggle.com/datasets/prachi13/customer-analytics/data)에서 제공됩니다.

---

<p align="center">
  <b>🚀 배송의 민족 팀 — 2026</b>
</p>
