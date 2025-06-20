from typing import Dict, Any, List, Callable
from sklearn.model_selection import KFold
import numpy as np
from .metrics import MultiMetricEvaluator


class ExperimentRunner:
    """Главный класс для автоматизации экспериментов по тюнингу генеративных моделей."""
    
    def __init__(self, 
                 dataset, 
                 model, 
                 tuner: None, 
                 hyperparameter_space: Dict[str, Any],
                 metrics: List[str], 
                 metric_weights: Dict[str, float] = None, 
                 n_folds: int = 5):
        """
        Инициализация экспериментального окружения.
        
        Args:
            dataset: объект датасета
            model: генеративная модель
            tuner: алгоритм тюнинга
            hyperparameter_space: пространство поиска гиперпараметров
            metrics: список метрик для оценки
            metric_weights: веса метрик
            n_folds: количество фолдов для кросс-валидации
        """
        self.dataset = dataset
        self.model = model
        self.tuner = tuner
        self.hyperparameter_space = hyperparameter_space
        self.metrics = metrics
        self.n_folds = n_folds
        
        # Устанавливаем веса метрик
        if metric_weights is None:
            # Равные веса для всех метрик
            self.metric_weights = {metric: 1.0 for metric in metrics}
        else:
            self.metric_weights = metric_weights
        
        # Для хранения результатов
        self.experiment_results = {}
        
    def run_experiment(self) -> Dict[str, Any]:
        """
        Запуск полного цикла оптимизации и оценки в заданном пространстве.
        
        Returns:
            Словарь с результатами эксперимента
        """
        
        # Создаем objective function
        objective_function = self.create_objective_function()
        
        # Запускаем оптимизацию
        best_hyperparameters, best_score, best_com_score = self.tuner.fit(
            objective_function, 
            self.hyperparameter_space
        )
        
        # Собираем результаты эксперимента
        self.experiment_results = {
            'best_hyperparameters': best_hyperparameters,
            'best_score': best_score,
            'best_com_score': best_com_score,
            'optimization_history': self.tuner.get_optimization_history(),
            'n_trials': self.tuner.n_trials,
            'hyperparameter_space': self.hyperparameter_space,
            'metrics_used': self.metrics,
            'metric_weights': self.metric_weights,
            'n_folds': self.n_folds
        }
        
        return self.experiment_results
    
    def create_objective_function(self) -> Callable:
        """
        Создание целевой функции для алгоритма тюнинга, которая использует кросс-валидацию.
        
        Returns:
            Функция для оптимизации
        """
        def objective(hyperparams: Dict[str, Any]) -> float:
            """
            Objective function для оптимизации (минимизация).
            
            Args:
                hyperparams: словарь с гиперпараметрами
                
            Returns:
                Значение loss (чем меньше, тем лучше)
            """
            evl_metrics = MultiMetricEvaluator(metrics=self.metrics, weights=self.metric_weights)
            
            model_copy = self._create_model_with_hyperparams(hyperparams)
            res, metrics = evl_metrics.evaluate(model_copy, self.dataset, n_folds=self.n_folds)
            
            return res, metrics
        
        return objective
    
    
    def _create_model_with_hyperparams(self, hyperparams: Dict[str, Any]):
        """
        Создание копии модели с заданными гиперпараметрами.
        
        Args:
            hyperparams: словарь гиперпараметров
            
        Returns:
            Модель с установленными гиперпараметрами
        """
        # Создаем копию модели
        model_copy = self.model.__class__(**hyperparams)
        return model_copy
    
        