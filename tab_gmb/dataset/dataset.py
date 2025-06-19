import pandas as pd
import os
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path


class Dataset:
    """
    Единый интерфейс для работы с табличными датасетами, 
    содержащий всю необходимую метаинформацию для генеративных моделей.
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        name: str, 
        task_type: str, 
        target_column: str, 
        num_columns: List[str], 
        cat_columns: List[str],
        verbose_init: bool =False
    ):
        """
        Инициализация объекта датасета с валидацией.
        
        Args:
            data: Табличные данные
            name: Название датасета
            task_type: Тип задачи ('classification' или 'regression')
            target_column: Название целевой колонки
            num_columns: Список числовых колонок
            cat_columns: Список категориальных колонок
        """
        self.data = data.copy()
        self.name = name
        self.task_type = task_type
        self.target_column = target_column
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.verbose_init = verbose_init
        
        # Валидация входных данных
        self._validate_dataset()
    
    def _validate_dataset(self) -> None:
        """Проверка корректности датасета и метаинформации."""
        
        # Проверка типа задачи
        if self.task_type not in ['classification', 'regression']:
            raise ValueError(f"task_type должен быть 'classification' или 'regression', получен: {self.task_type}")
        
        # Проверка наличия целевой колонки
        if self.target_column not in self.data.columns:
            raise ValueError(f"Целевая колонка '{self.target_column}' не найдена в данных")
        
        # Проверка числовых колонок
        missing_num_cols = set(self.num_columns) - set(self.data.columns)
        if missing_num_cols:
            raise ValueError(f"Числовые колонки не найдены в данных: {missing_num_cols}")
        
        # Проверка категориальных колонок
        missing_cat_cols = set(self.cat_columns) - set(self.data.columns)
        if missing_cat_cols:
            raise ValueError(f"Категориальные колонки не найдены в данных: {missing_cat_cols}")
        
        # Проверка пересечения числовых и категориальных колонок
        overlap = set(self.num_columns) & set(self.cat_columns)
        if overlap:
            raise ValueError(f"Колонки не могут быть одновременно числовыми и категориальными: {overlap}")
        
        # Проверка покрытия всех колонок (кроме целевой)
        feature_columns = set(self.num_columns) | set(self.cat_columns)
        all_columns = set(self.data.columns)
        uncovered = all_columns - feature_columns - {self.target_column}
        if uncovered:
            print(f"Предупреждение: Колонки не классифицированы как числовые или категориальные: {uncovered}")
        
        # Проверка типов данных в числовых колонках
        for col in self.num_columns:
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                print(f"Предупреждение: Колонка '{col}' помечена как числовая, но имеет не числовой тип")
        
        # Проверка пустого датасета
        if len(self.data) == 0:
            raise ValueError("Датасет не может быть пустым")
        
        if self.verbose_init:
            print(f"Датасет '{self.name}' успешно валидирован:")
            print(f"  - Размер: {self.data.shape}")
            print(f"  - Тип задачи: {self.task_type}")
            print(f"  - Целевая колонка: {self.target_column}")
            print(f"  - Числовые колонки ({len(self.num_columns)}): {self.num_columns}")
            print(f"  - Категориальные колонки ({len(self.cat_columns)}): {self.cat_columns}")
    
    
    def get_data_by_features(self, include_target: bool = False) -> pd.DataFrame:
        """
        Получение признаков без целевой переменной.
        
        Args:
            include_target: Включить ли целевую переменную
            
        Returns:
            DataFrame с признаками
        """
        feature_columns = self.num_columns + self.cat_columns
        if include_target:
            feature_columns.append(self.target_column)
        
        return self.data[feature_columns].copy()
    
    def get_target(self) -> pd.Series:
        """
        Получение целевой переменной.
        
        Returns:
            Series с целевой переменной
        """
        return self.data[self.target_column].copy()
    


if __name__ == "__main__":

    base_dir = os.path.dirname(__file__)
    test_dir = os.path.join(base_dir, 'insurance.csv')
    test_data = pd.read_csv(test_dir)

    dataset = Dataset(
        data=test_data,
        name='insurance',
        task_type='regression',
        target_column='expenses',
        num_columns=['age', 'bmi', 'children'],
        cat_columns=['sex', 'smoker', 'region'],
    )

    print(dataset.get_data_by_features())
    print(dataset.get_target())

    print(dataset.data.head())


    test_dir_2 = os.path.join(base_dir, 'titanic.csv')
    test_data_2 = pd.read_csv(test_dir_2)

    dataset_2 = Dataset(
        data=test_data_2,
        name='titanic',
        task_type='classification',
        target_column='Survived',
        num_columns=["Age", "Fare", "SibSp", "Parch"],
        cat_columns=["Pclass", "Sex", "Embarked"],
        verbose_init=True
    )

    print(dataset_2.get_data_by_features())
    print(dataset_2.get_target())

    print(dataset_2.data.head())

    
    


