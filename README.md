# Check Up Stroke: 건강검진 데이터 기반 뇌졸중 위험도 예측

> 2023년도 의료데이터 시각화 경진대회 우수상 수상 (Intel Korea) 

## 1. 📖 프로젝트 소개 (About)

**Check Up Stroke**는 건강검진 데이터를 기반으로 사용자의 뇌졸중 발병 위험도를 예측하는 인공지능 모델 프로젝트입니다.

뇌졸중은 전 세계 사망률 2위의 질병이지만, 높은 검사 비용 등으로 인해 조기 발견이 어렵습니다. 본 프로젝트는 비교적 접근성이 좋은 일반 건강검진 데이터를 활용하여 뇌졸중 위험 요소를 분석하고, 위험도를 예측하여 조기 예방에 기여하는 것을 목표로 합니다.

* **주요 기술**: `Python`, `Pandas`, `Scikit-learn`, `Power BI`
* **핵심 모델**: `Logistic Regression`
* **관련 자료**: [23학년도 의료데이터 시각화 경진대회 발표자료 (PDF)](./23학년도%20의료데이터%20시각화%20경진대회%20ppt.pdf)
* **데이터 출처**: [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

## 2. 🚀 주요 기능 (Features)

1.  **데이터 분석 및 피처 선정**:
    * **Power BI**를 활용하여 건강검진 데이터와 뇌졸중 간의 상관관계를 시각적으로 분석하고, 이를 바탕으로 주요 예측 변수(연령, 고혈압, BMI, 흡연 여부 등)를 선정합니다.
2.  **데이터 불균형 해소**:
    * **SMOTE (Synthetic Minority Over-sampling Technique)** 기법을 적용하여 뇌졸중 환자 데이터 부족(Class Imbalance) 문제를 해결하고 모델 학습 안정성을 확보합니다.
3.  **위험도 예측 모델**:
    * 전처리된 데이터를 바탕으로 **로지스틱 회귀(Logistic Regression)** 모델을 학습시켜 뇌졸중 발병 위험도를 예측합니다.
4.  **모델 성능 평가**:
    * 학습된 모델의 성능을 검증하기 위해 **ROC 커브** 및 **AUC 스코어**를 산출하고 평가합니다. (Test AUC 0.80 달성)
5.  **결과 시각화**:
    * 예측된 뇌졸중 위험도 결과를 시각화하여, 검진자가 자신의 위험 수준을 직관적으로 이해할 수 있도록 돕습니다.

## 3. 📊 모델 및 성능 (Model & Performance)

* **사용 모델**: 로지스틱 회귀 (Logistic Regression)
* **평가 지표**: ROC Curve, AUC (Area Under the Curve)
* **성능 결과**: 
    * **Train Data AUC**: 0.77
    * **Test Data AUC**: 0.80 



## 4. 👩‍💻 담당 업무 (My Contributions)

본 프로젝트에서 다음과 같은 역할을 주도적으로 수행했습니다.

* **데이터 전처리**:
     * 결측치(BMI 등) 처리 및 범주형 변수(성별, 흡연 여부 등)에 대한 One-Hot Encoding 수행.
     * **SMOTE** 알고리즘을 도입하여 뇌졸중 데이터의 클래스 불균형 문제 해결 및 모델 학습 안정성 확보.
* **피처 엔지니어링**: **Power BI**를 활용한 시각적 데이터 탐색(EDA)과 상관관계 분석을 통해 연령, 고혈압, BMI 등 주요 예측 변수 선정.
* **모델링 및 평가 보조**:
    * `Scikit-learn`과 `Imbalanced-learn` 파이프라인을 활용하여 전처리-학습-평가 과정을 자동화.
    * **로지스틱 회귀** 모델 학습 및 ROC 커브 기반 성능 평가 (Test AUC 0.80 달성).
* **데이터 시각화**: 분석 중간 과정과 최종 예측 모델의 결과를 시각화하여 의료적 인사이트 도출.

## 5. 💡 실행 방법 (How to Run)

1.  **데이터셋 준비:**
    * [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)에서 `healthcare-dataset-stroke-data.csv` 파일을 다운로드합니다.
    * 프로젝트 폴더 내에 `data` 폴더를 생성하고, 다운로드한 파일을 위치시킵니다.

2.  **레포지토리 클론:**
    ```bash
    git clone [https://github.com/yunajeon/Check-Up-Stroke.git](https://github.com/yunajeon/Check-Up-Stroke.git)
    cd Check-Up-Stroke
    ```

3.  **필요 라이브러리 설치:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **주피터 노트북 실행:**
    ```bash
    jupyter notebook Stroke_Prediction_Analysis.ipynb
    ```

## 6. 📂 레포지토리 구조 (Directory)
```
Check-Up-Stroke/
│
├── Stroke_Prediction_Analysis.ipynb (핵심 분석 및 모델링 코드)
├── requirements.txt                         (필요 라이브러리)
└── README.md                                (프로젝트 소개)
```
