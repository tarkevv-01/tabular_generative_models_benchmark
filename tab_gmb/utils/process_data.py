import numpy as np
import pandas as pd


def process_data(dataset, num_columns, cat_columns, transformation_num_type='None', transformation_cat_type='None'):

    df_processed = dataset.copy()

    # Обработка числовых признаков
    if transformation_num_type == 'CDF':
        # CDF трансформация: преобразует значения в их эмпирическую функцию распределения
        for col in num_columns:
            # Правильная формула для эмпирической CDF
            df_processed[col] = (df_processed[col].rank(method='average') - 0.5) / len(df_processed)

    elif transformation_num_type == 'PLE_CDF':
        # PLE_CDF (Probability Logit Envelope CDF) - более сложная трансформация
        for col in num_columns:
            # Сначала применяем CDF
            cdf_values = (df_processed[col].rank(method='average') - 0.5) / len(df_processed)

            # Затем применяем logit трансформацию с небольшим сглаживанием
            # Избегаем 0 и 1 для предотвращения бесконечности в logit
            epsilon = 1e-6
            cdf_values = np.clip(cdf_values, epsilon, 1 - epsilon)

            # Логит трансформация: ln(p/(1-p))
            df_processed[col] = np.log(cdf_values / (1 - cdf_values))

    # Обработка категориальных признаков
    if transformation_cat_type == 'OHE':
        # One Hot Encoding для категориальных признаков
        for col in cat_columns:
            # Создаем dummy переменные с префиксом имени колонки
            dummy_df = pd.get_dummies(df_processed[col], prefix=col, dtype=int)

            # Удаляем исходную категориальную колонку
            df_processed = df_processed.drop(columns=[col])

            # Добавляем новые dummy колонки
            df_processed = pd.concat([df_processed, dummy_df], axis=1)

    return df_processed