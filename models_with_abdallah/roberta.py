import pandas as pd

from model import Model

model_name = "FacebookAI/roberta-base"
roberta_model = Model(model_name)

abdallah_classification = pd.read_csv("../abdallah.csv")
abdallah_classification.dropna(inplace=True)
abdallah_classification = abdallah_classification.rename(columns={'Category': 'category', 'Message': 'sentence'})
labels = ["ham", "spam"]
roberta_model.train_evaluate(labels, abdallah_classification)

# Results:
# {'eval_loss': 0.06092388182878494,
# 'eval_accuracy': 0.9880382775119617,
# 'eval_f1': 0.9879131723548255,
# 'eval_runtime': 44.4428,
# 'eval_samples_per_second': 37.621,
# 'eval_steps_per_second': 2.363,
# 'epoch': 2.0}

# dataset url: https://www.kaggle.com/datasets/abdallahwagih/spam-emails