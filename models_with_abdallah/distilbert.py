import pandas as pd

from model import Model

model_name = "distilbert/distilbert-base-uncased"
distilbert_model = Model(model_name)

abdallah_classification = pd.read_csv("../abdallah.csv")
abdallah_classification.dropna(inplace=True)
abdallah_classification = abdallah_classification.rename(columns={'Category': 'category', 'Message': 'sentence'})
labels = ["ham", "spam"]
distilbert_model.train_evaluate(labels, abdallah_classification)

# Results:
# {'eval_loss': 0.05435454472899437,
# 'eval_accuracy': 0.9880382775119617,
# 'eval_f1': 0.9878476783268565,
# 'eval_runtime': 25.8538,
# 'eval_samples_per_second': 64.671,
# 'eval_steps_per_second': 4.061,
# 'epoch': 2.0}

# dataset url: https://www.kaggle.com/datasets/abdallahwagih/spam-emails