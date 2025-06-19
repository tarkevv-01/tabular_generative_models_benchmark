from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import sys, os
from .base_model import BaseGenerativeModel
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader


class TabDDPMModel(BaseGenerativeModel):
    """
    Обертка для модели TabDDPM из библиотеки SynthCity.
    
    TabDDPM (Tabular Denoising Diffusion Probabilistic Model) - это модель диффузии,
    адаптированная для генерации табличных данных.
    """
    
    def __init__(self, **kwargs):
        """
        Инициализация TabDDPM модели.
        
        Args:
            **kwargs: Гиперпараметры модели
        """
        super().__init__(**kwargs)
        
        # Установка значений по умолчанию
        default_params = {
            'batch_size': 4096,
            'lr': 7e-4,
            'num_timesteps': 1000,
            'n_layers_hidden': 4,
            'n_units_hidden': 512,
            'transformation_num_type': 'None',
            'transformation_cat_type': 'None',
            'n_iter': 1000,
            'device': 'cpu',
            'patience': 50
        }
        
        # Обновляем значения по умолчанию переданными параметрами
        for key, value in default_params.items():
            if key not in self.hyperparameters:
                self.hyperparameters[key] = value
        
        self._synthesizer = None
        self._dataloader = None
    
    def fit(self, data: pd.DataFrame, 
            target_column: Optional[str] = None,
            num_columns: Optional[List[str]] = None,
            cat_columns: Optional[List[str]] = None,
            **kwargs) -> None:
        """
        Обучение TabDDPM модели на предоставленных данных.
        
        Args:
            data: Табличные данные для обучения
            target_column: Название целевой колонки
            num_columns: Список числовых колонок
            cat_columns: Список категориальных колонок
            **kwargs: Дополнительные параметры обучения
        """
        def suppress_all_output():
            """Контекстный менеджер для подавления всего вывода"""
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    
        try:
            # Создаем dataloader для SynthCity
            self._dataloader = GenericDataLoader(data)
            
            # Подготавливаем параметры для TabDDPM
            ddpm_params = self._prepare_ddpm_params()
            
            # Создаем синтезатор
            self._synthesizer = Plugins().get(
                "tab_ddpm",
                **ddpm_params
            )
            
            print(f"Начинается обучение TabDDPM на {len(data)} образцах...")
            
            # Обучаем модель
            with suppress_all_output():
                self._synthesizer.fit(self._dataloader)
            
            self.is_fitted = True
            print("Обучение TabDDPM завершено успешно.")
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при обучении TabDDPM: {str(e)}")
    
    
    
    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Генерация синтетических данных.
        
        Args:
            n_samples: Количество генерируемых образцов
            
        Returns:
            DataFrame с синтетическими данными
        """
        
        try:
            # Генерируем данные через SynthCity
            synthetic_data = self._synthesizer.generate(count=n_samples)
            
            # Преобразуем обратно в pandas DataFrame
            if hasattr(synthetic_data, 'dataframe'):
                return synthetic_data.dataframe()
            else:
                return synthetic_data
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при генерации данных TabDDPM: {str(e)}")
        
    
    
    def _prepare_ddpm_params(self) -> Dict[str, Any]:
        """
        Подготовка параметров для TabDDPM синтезатора.
        
        Returns:
            Словарь параметров для TabDDPM из SynthCity
        """
        # Параметры, которые напрямую передаются в TabDDPM
        ddpm_params = {}
        
        # Соответствие наших параметров параметрам SynthCity TabDDPM
        param_mapping = {
            'batch_size': 'batch_size',
            'lr': 'lr',
            'num_timesteps': 'num_timesteps',
            'n_layers_hidden': 'n_layers_hidden',
            'n_units_hidden': 'n_units_hidden',
            'transformation_num_type': 'transformation_num_type',
            'transformation_cat_type': 'transformation_cat_type',
            'n_iter': 'n_iter',
            'device': 'device',
            'patience': 'patience'
        }
        
        for our_param, synthcity_param in param_mapping.items():
            if our_param in self.hyperparameters:
                ddpm_params[synthcity_param] = self.hyperparameters[our_param]
        
        return ddpm_params