from hyperopt import hp, fmin, tpe, rand, Trials, STATUS_OK
from .base import BaseTuner
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Tuple, List
from hyperopt import space_eval
import random
import numpy as np


class TPETuner(BaseTuner):
    """Реализация алгоритма случайного поиска в заданном пространстве параметров."""
    
    def __init__(self, n_trials: int = 50, random_state: int = None, **kwargs):
        """
        Инициализация Random Search алгоритма.
        
        Args:
            n_trials: количество итераций оптимизации
            random_state: seed для воспроизводимости
            **kwargs: дополнительные параметры
        """
        super().__init__(n_trials, random_state, **kwargs)
        self.trials = None
        self.RANDOM_SEED = random_state
    
    def fit(self, objective_function: Callable, hyperparameter_space: Dict[str, Any], verbose=False) -> Tuple[Dict[str, Any], float]:
        """
        Оптимизация с использованием случайного поиска через hyperopt.
        
        Args:
            objective_function: функция для оптимизации (минимизация)
            hyperparameter_space: пространство поиска гиперпараметров
            
        Returns:
            Кортеж (лучшие_параметры, лучшее_значение)
        """
        # Преобразуем пространство параметров в формат hyperopt
        hyperopt_space = self._convert_space_to_hyperopt(hyperparameter_space)
        
        # Создаем объект для хранения истории trials
        self.trials = Trials()
        
        # Wrapper для objective function
        def hyperopt_objective(params):
            try:
                # Вызываем исходную objective function
                loss, com_loss = objective_function(params)
                print('-'*50)
                print(params)
                print('Score:', loss)
                print(com_loss)
                print('-'*50)

                # Сохраняем в историю
                self.optimization_history.append({
                    'params': params.copy(),
                    'loss': loss,
                    'com_loss': com_loss,
                    'trial_number': len(self.optimization_history) + 1
                })
                
                return {'loss': loss, 'com_loss':com_loss, 'status': STATUS_OK}
            except Exception as e:
                return {'loss': float('inf'), 'com_loss':{}, 'status': STATUS_OK}
        
        # Запускаем оптимизацию
        rng = np.random.default_rng(self.RANDOM_SEED)
        best = fmin(
            fn=hyperopt_objective,
            space=hyperopt_space,
            algo=tpe.suggest,  # Используем TPE
            max_evals=self.n_trials,
            trials=self.trials,
            rstate=rng
        )
        best_params = space_eval(hyperopt_space, best)
        # Получаем лучшее значение
        best_trial = min(self.trials, key=lambda x: x['result']['loss'])
        best_loss = best_trial['result']['loss']
        best_com_loss = best_trial['result']['com_loss']
        
        return best_params, best_loss, best_com_loss
    
    def _convert_space_to_hyperopt(self, hyperparameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Преобразование пространства параметров в формат hyperopt.
        
        Args:
            hyperparameter_space: исходное пространство параметров
            
        Returns:
            Пространство в формате hyperopt
        """
        hyperopt_space = {}
        
        for param_name, param_config in hyperparameter_space.items():
            if isinstance(param_config, list):
                # Дискретные значения
                hyperopt_space[param_name] = hp.choice(param_name, param_config)
            elif isinstance(param_config, tuple):
                if len(param_config) == 2:
                    # Непрерывные без квантования
                    min_val, max_val = param_config
                    hyperopt_space[param_name] = hp.uniform(param_name, min_val, max_val)
                elif len(param_config) == 3:
                    # Непрерывные с квантованием
                    min_val, max_val, step = param_config
                    hyperopt_space[param_name] = hp.quniform(param_name, min_val, max_val, step)
        
        return hyperopt_space