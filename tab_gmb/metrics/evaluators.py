from typing import List
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sdmetrics.single_column import KSComplement
import warnings

def evaluate_c2st(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """
    Оценка различимости реальных и синтетических данных.
    
    Args:
        real_data: Реальные данные
        synthetic_data: Синтетические данные
        
    Returns:
        ROC-AUC score (0.5 - идеально, 1.0 - плохо)
    """
    real_copy = real_data.copy()
    if hasattr(synthetic_data, "dataframe"):
        synthetic_data = synthetic_data.dataframe()

    synthetic_copy = synthetic_data.copy()

    # Добавление меток
    real_copy['label'] = 1
    synthetic_copy['label'] = 0

    # Объединение данных
    df = pd.concat([real_copy, synthetic_copy], ignore_index=True)

    # Определение категориальных колонок
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_columns = [col for col in categorical_columns if col != 'label']

    # Label Encoding для категориальных признаков
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Разделение на признаки и метки
    X = df.drop(columns='label')
    y = df['label']

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    # Обучение XGBoost
    clf = XGBClassifier(
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Предсказания
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    return roc_auc_score(y_test, y_pred_proba)


def evaluate_ml_efficacy(real_train: pd.DataFrame, 
                        synthetic_train: pd.DataFrame, 
                        real_test: pd.DataFrame, 
                        cat_columns: List[str],
                        target_column: str, 
                        task_type: str) -> float:
    """
    Оценка практической применимости синтетических данных.
    
    Args:
        real_train: Реальные тренировочные данные
        synthetic_train: Синтетические тренировочные данные
        real_test: Реальные тестовые данные
        target_column: Название целевой колонки
        task_type: Тип задачи ('classification'/'regression')
        
    Returns:
        Отношение производительности (1.0 - идеально)
    """
        # Подготовка данных
    real_train_copy = real_train.copy()

    # Обработка синтетических данных (если это объект с методом dataframe)
    if hasattr(synthetic_train, "dataframe"):
        synthetic_train = synthetic_train.dataframe()
    synthetic_train_copy = synthetic_train.copy()

    real_test_copy = real_test.copy()

    # Определение категориальных колонок (исключая целевую)
    categorical_columns = cat_columns

    # Label Encoding для категориальных признаков
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        # Обучаем на реальных данных
        combined_values = pd.concat([real_train_copy[col], real_test_copy[col]]).astype(str)
        le.fit(combined_values)
        label_encoders[col] = le

        # Применяем к реальным данным
        real_train_copy[col] = le.transform(real_train_copy[col].astype(str))
        real_test_copy[col] = le.transform(real_test_copy[col].astype(str))

        # Применяем к синтетическим данным, обрабатываем неизвестные значения
        synthetic_values = synthetic_train_copy[col].astype(str)
        known_values = set(le.classes_)
        synthetic_values = synthetic_values.apply(lambda x: x if x in known_values else le.classes_[0])
        synthetic_train_copy[col] = le.transform(synthetic_values)

    # Разделение на признаки и целевую переменную
    X_real_train = real_train_copy.drop(columns=[target_column])
    y_real_train = real_train_copy[target_column]

    X_synthetic_train = synthetic_train_copy.drop(columns=[target_column])
    y_synthetic_train = synthetic_train_copy[target_column]

    X_real_test = real_test_copy.drop(columns=[target_column])
    y_real_test = real_test_copy[target_column]

    # Выбор модели в зависимости от типа задачи
    if task_type == 'classification':
        model_real = XGBClassifier(random_state=42, eval_metric='logloss')
        model_synthetic = XGBClassifier(random_state=42, eval_metric='logloss')
        score_func = f1_score
        score_kwargs = {'average': 'weighted'} if len(np.unique(y_real_train)) > 2 else {'average': 'binary'}
    else:  # regression
        model_real = XGBRegressor(random_state=42)
        model_synthetic = XGBRegressor(random_state=42)
        score_func = r2_score
        score_kwargs = {}

    try:
        # Обучение модели на реальных данных
        model_real.fit(X_real_train, y_real_train)
        y_pred_real = model_real.predict(X_real_test)
        score_real = score_func(y_real_test, y_pred_real, **score_kwargs)

        # Обучение модели на синтетических данных
        model_synthetic.fit(X_synthetic_train, y_synthetic_train)
        y_pred_synthetic = model_synthetic.predict(X_real_test)
        score_synthetic = score_func(y_real_test, y_pred_synthetic, **score_kwargs)

        # Вычисление ML-Efficacy
        if score_real > 0:
            ml_efficacy = score_synthetic / score_real
        else:
            ml_efficacy = 0.0

        return ml_efficacy

    except Exception as e:
        print(f"Ошибка при вычислении ML-Efficacy: {e}")
        return 0.0


def evaluate_pair(real_data: pd.DataFrame, 
                 synthetic_data: pd.DataFrame, 
                 cat_columns: List[str]) -> float:
    """
    Оценка сохранения корреляционных связей между признаками.
    
    Args:
        real_data: Реальные данные
        synthetic_data: Синтетические данные
        cat_columns: Список категориальных колонок
        
    Returns:
        Коэффициент сохранения корреляций (1.0 - идеально)
    """
    # One-hot энкодинг для категориальных признаков
    real_data_encoded = pd.get_dummies(real_data, columns=cat_columns, drop_first=True)

    if hasattr(synthetic_data, "dataframe"):
        synthetic = synthetic_data.dataframe()
    else:
        synthetic = synthetic_data

    synthetic_encoded = pd.get_dummies(synthetic, columns=cat_columns, drop_first=True)

    # После get_dummies
    real_data_encoded = real_data_encoded.astype(float)
    synthetic_encoded = synthetic_encoded.astype(float)

    # Корреляции
    corr_real = real_data_encoded.corr()
    corr_synth = synthetic_encoded.corr()

    # Абсолютное отклонение корреляций
    pairwise_diff = np.abs(corr_real - corr_synth)

    # Усреднённое отклонение
    mean_diff = pairwise_diff.mean().mean()

    # Проверяем на NaN
    if np.isnan(mean_diff):
        print("Warning: mean_diff is NaN, returning 0")
        return 0.0

    return 1 - mean_diff


def evaluate_shape(real_data: pd.DataFrame, 
                  synthetic_data: pd.DataFrame, 
                  cat_columns: List[str]) -> float:
    """
    Оценка сохранения индивидуальных распределений признаков.
    
    Args:
        real_data: Реальные данные
        synthetic_data: Синтетические данные
        cat_columns: Список категориальных колонок
        
    Returns:
        Коэффициент сохранения распределений (1.0 - идеально)
    """
    # Обработка входных данных
    if hasattr(synthetic_data, "dataframe"):
        synthetic = synthetic_data.dataframe()
    else:
        synthetic = synthetic_data.copy()

    real_data = pd.get_dummies(real_data, columns=cat_columns, drop_first=True)
    synthetic = pd.get_dummies(synthetic, columns=cat_columns, drop_first=True)

    # После get_dummies
    real_data = real_data.astype(float)
    synthetic = synthetic.astype(float)

    # Пытаемся использовать прямые метрики
    shape_scores = []

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for column in real_data.select_dtypes(include=['number']).columns:
            if column in synthetic.columns:
                ks_score = KSComplement.compute(
                    real_data=real_data[column],
                    synthetic_data=synthetic[column]
                )
                shape_scores.append(ks_score)

    return sum(shape_scores) / len(shape_scores) if shape_scores else 0.0