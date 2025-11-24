# Intro to AI / Machine Learning Basic  
### 과제: Classification / Regression / K-means 구현

본 저장소는 **인공지능개론(Introduction to AI)** 과목에서 수행한 세 가지 머신러닝 실습 과제를 정리한 프로젝트입니다.

- **1) Classification (MLP 기반 이진 분류)**
- **2) Regression (MLP 기반 회귀)**
- **3) K-means Clustering (비지도 학습)**

Python, PyTorch, Scikit-learn 등을 사용하여 모델을 구현하였으며, 각 실습별로 데이터 전처리, 모델 구성, 학습, 평가 과정이 포함되어 있습니다.


# 1. Classification — Titanic Survival Prediction  
MLP 기반의 간단한 이진 분류 문제로, Titanic 데이터셋을 사용했습니다.

### 주요 내용
- Train/Validation/Test 분리  
- PyTorch MLP 모델 구현  
- CrossEntropyLoss & Adam Optimizer  
- 정확도(Accuracy) 기반 평가

### 결과 (요약)
- **최종 Test Accuracy: 약 70%**
- 단순 전처리만 적용한 베이스라인 모델 기준


# 2. Regression — House Price Prediction  
Ames Housing Dataset(일부 feature 선정)을 활용하여 집값을 예측하는 회귀 실습입니다.

### 주요 내용
- Numeric Feature 선택  
- Median 기반 결측치 처리  
- StandardScaler 정규화  
- MLP 기반 회귀 모델 구현  
- MSE / RMSE 평가

### 결과 (요약)
- **Final Test RMSE ≈ 175,000**
- Feature engineering 없이 baseline 성능


# 3. K-means Clustering — Mall Customers Dataset  
Mall Customers 데이터를 활용하여 고객을 소비 패턴 기준으로 군집화한 실습입니다.

### 주요 내용
- K = 5 클러스터 설정  
- 연소득 & 소비 점수만 사용하여 2D 클러스터링  
- Standard normalization 적용  
- 무작위 초기화 → 반복적 E-step / M-step 수행  
- matplotlib을 통한 클러스터링 과정 시각화



## 기술 스택

- **Python 3**
- **PyTorch**
- **pandas / numpy**
- **matplotlib**
- **scikit-learn**
  
