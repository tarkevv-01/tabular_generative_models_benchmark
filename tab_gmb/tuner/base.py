from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Tuple, List
import random
import numpy as np


class BaseTuner(ABC):
    """Абстрактный базовый класс для всех алгоритмов тюнинга."""
    
    def __init__(self, n_trials: int = 50, random_state: int = 42, **kwargs):
        """
        Инициализация алгоритма тюнинга.
        
        Args:
            n_trials: количество итераций оптимизации
            random_state: seed для воспроизводимости
            **kwargs: специфичные для алгоритма параметры
        """
        self.n_trials = n_trials
        self.random_state = random_state
        self.optimization_history = []
        
        # Устанавливаем seed для воспроизводимости
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Сохраняем дополнительные параметры
        self.kwargs = kwargs
    
    @abstractmethod
    def fit(self, objective_function: Callable, hyperparameter_space: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Основной метод оптимизации в заданном пространстве параметров.
        
        Args:
            objective_function: функция для оптимизации
            hyperparameter_space: пространство поиска гиперпараметров
            
        Returns:
            Кортеж (лучшие_параметры, лучшее_значение)
        """
        pass
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Получение истории поиска для анализа.
        
        Returns:
            История оптимизации
        """
        return self.optimization_history.copy()
    
    def _sample_hyperparameters(self, hyperparameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Сэмплирование гиперпараметров из заданного пространства.
        
        Args:
            hyperparameter_space: пространство поиска
            
        Returns:
            Словарь с сэмплированными параметрами
        """
        sampled_params = {}
        
        for param_name, param_config in hyperparameter_space.items():
            if isinstance(param_config, list):
                # Дискретные значения
                sampled_params[param_name] = random.choice(param_config)
            elif isinstance(param_config, tuple):
                if len(param_config) == 2:
                    # Непрерывные без квантования
                    min_val, max_val = param_config
                    sampled_params[param_name] = random.uniform(min_val, max_val)
                elif len(param_config) == 3:
                    # Непрерывные с квантованием
                    min_val, max_val, step = param_config
                    n_steps = int((max_val - min_val) / step)
                    step_idx = random.randint(0, n_steps)
                    sampled_params[param_name] = min_val + step_idx * step
        
        return sampled_params