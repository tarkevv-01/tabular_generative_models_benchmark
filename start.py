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
    verbose_init=True
)

print(dataset.get_data_by_features())
print(dataset.get_target())

# print(dataset.data.head())
# ctgan = CTGANModel()
# print(ctgan.get_hyperparameters())


model_gm = TabDDPMModel(**{'batch_size': 4096, 'lr': 0.0007000000000000001, 'n_layers_hidden': 4, 'n_units_hidden': 256, 'num_timesteps': 1000, 'transformation_cat_type': 'None', 'transformation_num_type': 'None'})
print(model_gm.get_hyperparameters())


model_gm.fit(
    data=dataset.get_data_by_features(),
    target_column=dataset.target_column,
    num_columns=dataset.num_columns,
    cat_columns=dataset.cat_columns,
)

synthetic_data = model_gm.generate(len(dataset.data))

print(synthetic_data.head())
print('c2st', evaluate_c2st(dataset.get_data_by_features(), synthetic_data))
print('pair', evaluate_pair(dataset.get_data_by_features(), synthetic_data, cat_columns=dataset.cat_columns))
print('shape', evaluate_shape(dataset.get_data_by_features(), synthetic_data, cat_columns=dataset.cat_columns))

# evl_metrics = MultiMetricEvaluator(metrics=['c2st'], weights={'c3st': 1.0})