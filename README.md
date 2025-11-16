# Check Up Stroke: 건강검진 데이터 기반 뇌졸중 위험도 예측

> 2023년도 의료데이터 시각화 경진대회 우수상 수상 (Intel Korea) 

## 1. 📖 프로젝트 소개 (About)

**Check Up Stroke**는 건강검진 데이터를 기반으로 사용자의 뇌졸중 발병 위험도를 예측하는 인공지능 모델 프로젝트입니다.

뇌졸중은 전 세계 사망률 2위의 질병이지만, 높은 검사 비용 등으로 인해 조기 발견이 어렵습니다. 본 프로젝트는 비교적 접근성이 좋은 일반 건강검진 데이터를 활용하여 뇌졸중 위험 요소를 분석하고, 위험도를 예측하여 조기 예방에 기여하는 것을 목표로 합니다.

* **주요 기술**: `Python`, `Pandas`, `Scikit-learn`, `Power BI`
* **핵심 모델**: `Logistic Regression` (로지스틱 회귀)
* **관련 자료**: [23학년도 의료데이터 시각화 경진대회 발표자료 (PDF)](./23학년도%20의료데이터%20시각화%20경진대회%20ppt.pdf)

## 2. 🚀 주요 기능 (Features)

1.  **데이터 전처리 및 분석**:
    * 건강검진 데이터에서 뇌졸중과 높은 상관관계를 가진 주요 변수(연령, 고혈압, BMI, 흡연 여부 등)를 분석하고 선정합니다.
2.  **위험도 예측 모델**:
    * 선정된 피처들을 바탕으로 **로지스틱 회귀(Logistic Regression)** 모델을 학습시켜 뇌졸중 발병 위험도를 예측합니다.
3.  **모델 성능 평가**:
    * **ROC 커브** 및 **AUC 스코어**를 사용하여 모델의 성능을 객관적으로 평가합니다.
4.  **결과 시각화 (Power BI)**:
    * 예측된 뇌졸중 위험도와 주요 변수(BMI, 연령, 고혈압, 흡연) 간의 관계를 **Power BI 대시보드**로 시각화하여 사용자가 직관적인 인사이트를 얻을 수 있도록 제공합니다.

## 3. 📊 모델 및 성능 (Model & Performance)

* **사용 모델**: 로지스틱 회귀 (Logistic Regression)
* **평가 지표**: ROC Curve, AUC (Area Under the Curve)
* **성능 결과**: 
    * **Train Data AUC**: 0.77
    * **Test Data AUC**: 0.80 



## 4. 👩‍💻 담당 업무 (My Contributions)

본 프로젝트에서 다음과 같은 역할을 주도적으로 수행했습니다.

* **데이터 전처리**: 원본 건강검진 데이터의 결측치 처리 및 모델 학습을 위한 데이터셋 정제.
* **피처 엔지니어링**: 뇌졸중과 상관관계가 높은 주요 변수(연령, 고혈압, BMI 등)를 선정하고 분석.
* **모델링 및 평가 보조**:
    * **로지스틱 회귀** 모델을 적용하여 뇌졸중 위험도 예측 모델 구현.
    * ROC 커브 분석을 통한 모델 성능 평가 (Test AUC 0.80 달성).
* **데이터 시각화**: **Power BI**를 활용하여 모델의 예측 결과와 주요 변수 간의 관계를 시각화하는 대시보드 제작.

## 5. 💡 실행 방법 (How to Run)

1.  **레포지토리 클론:**
    ```bash
    git clone https://github.com/yunajeon/Check-Up-Stroke.git
    cd Check-Up-Stroke
    ```

2.  **가상 환경 생성 및 활성화:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **필요 라이브러리 설치:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **모델 학습 스크립트 실행:**
    ```bash
    python src/stroke_prediction_model.py
    ```

## 6. 📂 레포지토리 구조 (Directory)
Check-Up-Stroke/
│
├── data/
│
├── notebooks/
│   └── 1_Data_Analysis_Visualization.ipynb  (데이터 분석 및 시각화 노트북)
│   └── 2_Model_Training.ipynb               (모델 학습 및 평가 노트북)
│
├── src/
│   └── stroke_prediction_model.py           (본 알고리즘 스크립트)
│
├── requirements.txt                         (필요 라이브러리)
└── README.md                                (프로젝트 소개)
