import os
import pandas as pd
from tab_gmb.dataset import Dataset
from tab_gmb.models import CTGANModel, TabDDPMModel
from tab_gmb.metrics import MultiMetricEvaluator

# base_dir = os.path.dirname(__file__)
# test_dir = os.path.join(base_dir, 'titanic.csv')
# test_data = pd.read_csv(test_dir)

# dataset = Dataset(
#     data=test_data,
#     name='titanic',
#     task_type='classification',
#     target_column='Survived',
#     num_columns=["Age", "Fare", "SibSp", "Parch"],
#     cat_columns=["Pclass", "Sex", "Embarked"],
#     verbose_init=True
# )

# print(dataset.get_data_by_features())
# print(dataset.get_target())

# print(dataset.data.head())
# ctgan = CTGANModel()
# print(ctgan.get_hyperparameters())


# ddmp = TabDDPMModel()
# print(ddmp.get_hyperparameters())

evl_metrics = MultiMetricEvaluator(metrics=['c2st'], weights={'c3st': 1.0})