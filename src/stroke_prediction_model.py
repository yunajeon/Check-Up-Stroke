import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def load_data(file_path):
    """데이터셋 로드"""
    print(f"데이터 로드 시도: {file_path}")
    try:
        data = pd.read_csv(file_path)
        print("데이터 로드 성공.")
        return data
    except FileNotFoundError:
        print("경고: 데이터 파일을 찾을 수 없습니다. 예시 데이터로 대체합니다.")
        print("프로젝트 진행 시 'data/your_dataset.csv' 경로에 실제 데이터를 위치시키세요.")
        return None

def preprocess_data(data):
    """
    데이터 전처리 및 피처 엔지니어링 수행
    """

    features = ['age', 'hypertension', 'bmi', 'gender', 'smoking_status']
    target = 'stroke'
    
    # 'Unknown' 값 등은 결측치로 처리
    data = data.replace('Unknown', pd.NA)
    data = data.dropna(subset=features + [target]) # 주요 피처 결측치 제거
    
    X = data[features]
    y = data[target]
    
    # 숫자형 피처와 범주형 피처 분리
    numeric_features = ['age', 'bmi']
    categorical_features = ['gender', 'smoking_status', 'hypertension'] # 고혈압(0, 1)도 범주형으로 처리
    
    # 전처리 파이프라인 구성
    # 숫자형: 결측치 중앙값 대체, 표준화
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 범주형: 결측치 'missing' 대체, 원핫인코딩
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # ColumnTransformer로 파이프라인 결합
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor

def train_and_evaluate(X, y, preprocessor):
    """
    로지스틱 회귀 모델을 학습하고 ROC 커브로 평가
    """
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 로지스틱 회귀 모델 파이프라인
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # 모델 학습
    model_pipeline.fit(X_train, y_train)
    
    # 예측 (확률)
    y_train_proba = model_pipeline.predict_proba(X_train)[:, 1]
    y_test_proba = model_pipeline.predict_proba(X_test)[:, 1]
    
    # AUC 스코어 계산 
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print("\n--- 모델 성능 평가 ---")
    print(f"Train AUC Score: {train_auc:.4f} (프로젝트 발표 자료 기준: 0.77 [cite: 257])")
    print(f"Test AUC Score: {test_auc:.4f} (프로젝트 발표 자료 기준: 0.80 [cite: 275])")
    
    # 분류 리포트 (Test)
    y_test_pred = model_pipeline.predict(X_test)
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    return y_train, y_train_proba, y_test, y_test_proba, train_auc, test_auc

def plot_roc_curve(y_train, y_train_proba, y_test, y_test_proba, train_auc, test_auc):
    """
    Train/Test ROC 커브 시각화
    """
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    
    plt.figure(figsize=(12, 6))
    
    # Train ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr_train, tpr_train, label=f'Classifier (AUC = {train_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--') # Random guess line
    plt.title('Train ROC Curve - stroke')
    plt.xlabel('False Positive Rate (Positive label: 1)')
    plt.ylabel('True Positive Rate (Positive label: 1)')
    plt.legend(loc='lower right')
    
    # Test ROC Curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr_test, tpr_test, label=f'Classifier (AUC = {test_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--') # Random guess line
    plt.title('Test ROC Curve - stroke')
    plt.xlabel('False Positive Rate (Positive label: 1)')
    plt.ylabel('True Positive Rate (Positive label: 1)')
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('roc_curves.png')
    print("\n'roc_curves.png' 파일로 ROC 커브가 저장되었습니다.")

def main():
    """메인 실행 함수"""
    DATA_FILE_PATH = 'data/health_checkup_data.csv'
    
    # 1. 데이터 로드
    data = load_data(DATA_FILE_PATH)
    
    if data is None:
        print("데이터를 로드할 수 없어 프로그램을 종료합니다.")
        return

    # 2. 데이터 전처리
    X, y, preprocessor = preprocess_data(data)
    
    # 3. 모델 학습 및 평가
    y_train, y_train_proba, y_test, y_test_proba, train_auc, test_auc = train_and_evaluate(X, y, preprocessor)
    
    # 4. ROC 커브 시각화
    plot_roc_curve(y_train, y_train_proba, y_test, y_test_proba, train_auc, test_auc)

if __name__ == "__main__":
    main()