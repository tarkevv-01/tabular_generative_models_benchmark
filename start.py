import os
import pandas as pd
from tab_gmb.dataset import Dataset
from tab_gmb.models import CTGANModel, TabDDPMModel
from tab_gmb.metrics import MultiMetricEvaluator, evaluate_c2st, evaluate_pair, evaluate_shape

base_dir = os.path.dirname(__file__)
test_dir = os.path.join(base_dir, 'titanic.csv')
test_data = pd.read_csv(test_dir)

dataset = Dataset(
    data=test_data,
    name='titanic',
    task_type='classification',
    target_column='Survived',
    num_columns=["Age", "Fare", "SibSp", "Parch"],
    cat_columns=["Pclass", "Sex", "Embarked"],
    verbose_init=False
)


model_gm_1= TabDDPMModel()
model_gm_2= CTGANModel()
evl_metrics = MultiMetricEvaluator(metrics=['c2st', 'pair'], weights={'c2st': 2.0, 'pair': -1.0})

print('Tab-DDPM')
print(evl_metrics.evaluate(model_gm_1, dataset, n_folds=3))

print('CTGAN')
print(evl_metrics.evaluate(model_gm_2, dataset, n_folds=3))