from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import sys, os
import builtins
import contextlib
from .base_model import BaseGenerativeModel
from synthcity.plugins import Plugins
from synthcity.plugins.generic.plugin_ddpm import TabDDPMPlugin as DDPM
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
            'batch_size': 1024,
            'lr': 0.002,
            'weight_decay': 1e-4,
            'num_timesteps': 1000,

            'n_layers_hidden': 3,  # Будет передано в model_params
            'n_units_hidden': 256, # Будет передано в model_params
            'dropout': 0.0,        # Будет передано в model_params
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
        @contextlib.contextmanager
        def suppress_all_output():
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                old_print = builtins.print
                builtins.print = lambda *args, **kwargs: None
                sys.stdout = devnull
                sys.stderr = devnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    builtins.print = old_print
                    
        try:
            # Создаем dataloader для SynthCity
            self._dataloader = GenericDataLoader(data)
            
            # Подготавливаем параметры для TabDDPM
            ddpm_params = self._prepare_ddpm_params()
            
            # Создаем синтезатор
            self._synthesizer = DDPM(
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
        
        # Параметры модели, которые должны быть в model_params
        model_params = {}
        
        # Соответствие наших параметров параметрам SynthCity TabDDPM
        direct_param_mapping = {
            'batch_size': 'batch_size',
            'lr': 'lr',
            'weight_decay': 'weight_decay',
            'num_timesteps': 'num_timesteps',
        }
        
        # Параметры, которые должны быть в model_params
        model_param_keys = {
            'n_layers_hidden',
            'n_units_hidden', 
            'dropout'
        }
        
        # Заполняем прямые параметры
        for our_param, synthcity_param in direct_param_mapping.items():
            if our_param in self.hyperparameters:
                ddpm_params[synthcity_param] = self.hyperparameters[our_param]
        
        # Заполняем model_params
        for param_key in model_param_keys:
            if param_key in self.hyperparameters:
                model_params[param_key] = self.hyperparameters[param_key]
        
        # Добавляем model_params в основные параметры
        if model_params:
            ddpm_params['model_params'] = model_params
        
        return ddpm_params