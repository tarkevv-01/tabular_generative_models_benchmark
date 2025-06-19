from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from .evaluators import evaluate_c2st, evaluate_ml_efficacy, evaluate_pair, evaluate_shape


class MultiMetricEvaluator:
    """
    Система многокритериальной оценки с поддержкой кросс-валидации.
    """
    
    def __init__(self, metrics: List[str], weights: Optional[Dict[str, float]] = None):
        """
        Инициализация системы оценки.
        
        Args:
            metrics: Список используемых метрик
            weights: Веса метрик для агрегации
        """
        self.metrics = metrics
        self.weights = weights or {metric: 1.0 for metric in metrics}
        
        # Проверяем, что все метрики поддерживаются
        supported_metrics = {'c2st', 'ml_efficacy', 'pair', 'shape'}
        for metric in metrics:
            if metric not in supported_metrics:
                raise ValueError(f"Неподдерживаемая метрика: {metric}")
        
        # Проверяем, что веса заданы для всех метрик
        for metric in metrics:
            if metric not in self.weights:
                self.weights[metric] = 1.0
    
    def evaluate(self, model, dataset, n_folds: int = 5) -> float:
        """
        Оценка модели с заданными гиперпараметрами через кросс-валидацию.
        
        Args:
            model: Генеративная модель
            hyperparameters: Гиперпараметры для тестирования
            dataset: Объект датасета
            n_folds: Количество фолдов для кросс-валидации
            
        Returns:
            Агрегированная метрика качества
        """
        
        # Инициализируем кросс-валидацию
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(dataset.data)):
            try:
                # Разделяем данные на train/val
                train_data = dataset.get_data_by_features(True).iloc[train_idx].copy()
                val_data = dataset.get_data_by_features(True).iloc[val_idx].copy()
                
                # Обучаем модель на train
                model.fit(
                    train_data, 
                    target_column=dataset.target_column,
                    num_columns=dataset.num_columns,
                    cat_columns=dataset.cat_columns
                )
                
                # Генерируем синтетические данные
                synthetic_data = model.generate(len(val_data))
                
                # Вычисляем метрики для этого фолда
                fold_metrics = self._compute_fold_metrics(
                    train_data, val_data, synthetic_data, dataset
                )
                
                fold_scores.append(fold_metrics)
                
            except Exception as e:
                print(f"Ошибка в фолде {fold_idx}: {str(e)}")
                # Возвращаем плохой скор в случае ошибки
                return float('inf')
        
        # Агрегируем результаты по фолдам
        avg_metrics = self._aggregate_fold_results(fold_scores)
        
        # Нормализуем и взвешиваем метрики
        #normalized_metrics = self.normalize_metrics(avg_metrics)
        
        # Вычисляем итоговый скор
        final_score = self._compute_weighted_score(avg_metrics)
        
        return final_score, avg_metrics
    
    def _compute_fold_metrics(self, train_data: pd.DataFrame, 
                             val_data: pd.DataFrame, 
                             synthetic_data: pd.DataFrame, 
                             dataset) -> Dict[str, float]:
        """
        Вычисление метрик для одного фолда.
        
        Args:
            train_data: Тренировочные данные
            val_data: Валидационные данные
            synthetic_data: Синтетические данные
            dataset: Объект датасета
            
        Returns:
            Словарь с метриками для фолда
        """
        metrics_values = {}
        
        for metric in self.metrics:
            cat_columns = dataset.cat_columns.copy()
            if dataset.task_type == 'classification':
                cat_columns.append(dataset.target_column)

            try:
                if metric == 'c2st':
                    metrics_values[metric] = evaluate_c2st(val_data, synthetic_data)
                    
                elif metric == 'ml_efficacy':
                    metrics_values[metric] = evaluate_ml_efficacy(
                        train_data, synthetic_data, val_data, 
                        cat_columns, dataset.target_column, dataset.task_type
                    )
                    
                elif metric == 'pair':
                    metrics_values[metric] = evaluate_pair(
                        val_data, synthetic_data, cat_columns 
                    )
                    
                elif metric == 'shape':
                    metrics_values[metric] = evaluate_shape(
                        val_data, synthetic_data, cat_columns 
                    )
                    
            except Exception as e:
                print(f"Ошибка при вычислении метрики {metric}: {str(e)}")
                # Устанавливаем плохое значение для проблемной метрики
                if metric == 'c2st':
                    metrics_values[metric] = 1.0  # Худшее значение для C2ST
                else:
                    metrics_values[metric] = 0.0  # Худшее значение для остальных
        
        return metrics_values
    
    def _aggregate_fold_results(self, fold_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Агрегация результатов по фолдам.
        
        Args:
            fold_scores: Список словарей с метриками по фолдам
            
        Returns:
            Словарь с усредненными метриками
        """
        if not fold_scores:
            return {metric: 0.0 for metric in self.metrics}
        
        avg_metrics = {}
        for metric in self.metrics:
            values = [fold[metric] for fold in fold_scores if metric in fold]
            if values:
                avg_metrics[metric] = np.mean(values)
            else:
                # Если метрика не была вычислена ни в одном фолде
                if metric == 'c2st':
                    avg_metrics[metric] = 1.0
                else:
                    avg_metrics[metric] = 0.0
        
        return avg_metrics
    
    def normalize_metrics(self, metrics_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Нормализация метрик для корректной агрегации.
        
        Args:
            metrics_dict: Словарь сырых значений метрик
            
        Returns:
            Словарь нормализованных метрик
        """
        normalized = {}
        
        for metric, value in metrics_dict.items():
            # if metric == 'c2st':
            #     # C2ST: 0.5 - идеально, 1.0 - плохо
            #     # Нормализуем к диапазону [0, 1], где 1 - лучше
            #     normalized[metric] = 2.0 * (1.0 - value)  # Теперь 1.0 - лучше, 0.0 - хуже
            # else:
            #     # Остальные метрики: 1.0 - идеально, 0.0 - плохо
            #     # Уже в правильном диапазоне
            #     normalized[metric] = value
            normalized[metric] = value
        
        return normalized
    
    def _compute_weighted_score(self, normalized_metrics: Dict[str, float]) -> float:
        """
        Вычисление взвешенного итогового скора.
        
        Args:
            normalized_metrics: Нормализованные метрики
            
        Returns:
            Взвешенный скор (чем больше, тем лучше)
        """
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for metric, value in normalized_metrics.items():
            #weight = abs(self.weights[metric])  # Используем абсолютное значение веса
            weight = self.weights[metric]
            total_weighted_score += weight * value
            total_weight += weight
        
        # if total_weight == 0:
        #     return 0.0
        
        # # Возвращаем негативный скор для минимизации (так как алгоритмы ищут минимум)
        # return -total_weighted_score / total_weight
        return total_weighted_score
    
    