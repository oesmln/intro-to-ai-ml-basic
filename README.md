# Intro to AI / Machine Learning Basic  
### 과제: Classification / Regression / K-means 구현

본 저장소는 **인공지능개론(Introduction to AI)** 과목에서 수행한 세 가지 머신러닝 실습 과제를 정리한 프로젝트입니다.

- **1) Classification – Titanic 생존 예측**
- **2) Regression – House Price 예측**
- **3) K-means Clustering – Mall Customers 군집화**

Python, PyTorch, Scikit-learn 등을 사용하여  
각 과제마다 **데이터 전처리 → 모델 설계 → 하이퍼파라미터 튜닝 → validation 기반 모델 선택 → test 평가 → 분석/시각화** 과정을 수행했습니다.


# 1. Classification — Titanic Survival Prediction  

Titanic 데이터셋을 사용한 **이진 분류 문제**입니다.  
승객의 나이, 성별, 객실 등 정보를 이용해 생존 여부(`Survived`)를 예측합니다.

## 1-1. 주요 내용

- Train / Validation / Test 분리
- 전처리
  - 불필요한 문자열 컬럼 제거 (`PassengerId`, `Name`, `Ticket`, `Cabin`)
  - `Sex`, `Embarked` 라벨 인코딩
  - `Age`, `Fare` 결측치 중앙값 대체
  - `StandardScaler`를 이용한 feature 스케일링
- 모델 구성
  - **PyTorch MLP**
    - 기본 구조: `64-64` hidden layer  
    - 확장 구조: `128-64 + Dropout(0.3)` 등 여러 구조 실험
    - 하이퍼파라미터 튜닝: `lr = 1e-3, 5e-4`, `epochs = 20, 30`, `batch_size = 32, 64` 등
  - **Scikit-learn 기반 모델**
    - `LogisticRegression`
    - `SVM (RBF kernel)`
    - `RandomForestClassifier`
    - `GradientBoostingClassifier`

## 1-2. 모델 선택 방식

- 모든 전략(여러 MLP + Logistic/SVM/RF/GB)을 학습 후  
  **Validation Accuracy** 기준으로 성능을 비교
- 상위 **Top 3 전략**을 선택하고, 각각에 대한 **Test Accuracy**를 함께 보고

예시 Top 3 전략:
- `mlp_128x64 + dropout 0.3`
- `mlp_64x64`
- `svm_rbf`

## 1-3. 결과 (요약)

- Validation 기준 Top 3 모델 모두 **약 0.82~0.83 수준의 val accuracy**  
- 해당 모델들의 test accuracy는 **약 0.84 수준**
- Extra 분석
  - MLP 학습 곡선(Train / Val Accuracy)
  - Confusion Matrix
  - Classification Report (precision/recall/F1)


# 2. Regression — House Price Prediction  

Ames Housing Dataset(일부 numeric feature만 사용)을 이용해  
주택 가격(`SalePrice`)을 예측하는 **회귀 문제**를 다룹니다.

## 2-1. 주요 내용

- Target: `SalePrice`
- 사용한 주요 feature:
  - `OverallQual`, `OverallCond`,
    `GrLivArea`, `TotalBsmtSF`,
    `YearBuilt`, `YearRemodAdd`,
    `1stFlrSF`, `2ndFlrSF`,
    `GarageCars`, `GarageArea`,
    `FullBath`, `HalfBath`,
    `TotRmsAbvGrd`, `Fireplaces`,
    `LotArea`
- 전처리
  - 결측치는 모두 **median**으로 처리
  - 입력 feature는 `StandardScaler()`로 정규화
  - Train / Val / Test = **6 : 2 : 2**

## 2-2. 모델 & 전략

### MLP 기반 회귀 모델
- 여러 구조 실험:
  - `mlp_64x64_lr1e-3_ep40_bs64`
  - `mlp_128x64_do0.2_lr1e-3_ep40_bs64`
  - `mlp_64_lr5e-4_ep60_bs32`
- Loss: MSE  
- Optimizer: Adam  
- Metrics: **MSE / MAE / RMSE**

### 기본 ML 회귀 모델
- `LinearRegression`
- `RandomForestRegressor`
- `GradientBoostingRegressor`

## 2-3. 모델 선택 & 결과

- Validation MSE 기준으로 **Top 3 전략 자동 선택**
- Test MSE / MAE 출력

요약:
- `RandomForestRegressor`, `GradientBoostingRegressor`가 가장 우수
- Best 모델 기준 Test RMSE ≈ **20,000**

## 2-4. Extra 분석

- **예측 vs 실제 산점도**
- **Residual vs Predicted 플롯**
- Residual 히스토그램
- Tree 모델의 **Feature Importance 시각화**
  - `OverallQual`, `GrLivArea`, `TotalBsmtSF` 중요도 높음


# 3. K-means Clustering — Mall Customers Dataset  

고객의 연소득 & 소비 점수를 기반으로 군집화하는 비지도 학습 실습입니다.

## 3-1. 주요 내용

- Feature:  
  - `Annual Income (k$)`  
  - `Spending Score (1-100)`
- Train / Validation / Test = 6:2:2
- Feature는 모두 `StandardScaler` 로 정규화

## 3-2. K-means 전략 구성

- k값 후보: **3, 4, 5, 6, 7**
- 초기화 방식:  
  - `init="random"`  
  - `init="k-means++"`
- 총 **10개 전략(k × init)** 을 학습

## 3-3. 모델 선택 & 결과

- Validation Silhouette Score 기준으로 성능 비교 후  
  **Top 3 전략 자동 정렬**
- Best 전략(`k=5`) test silhouette ≈ **0.53**

## 3-4. Extra 분석

- Best 모델 cluster 시각화 (산점도 + centroid X 표시)
- Silhouette Score Top10 bar chart
- Elbow Method(SSE vs k)


## 기술 스택

- Python 3  
- PyTorch  
- scikit-learn  
- pandas / numpy  
- matplotlib  
  
