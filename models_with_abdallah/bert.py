import pandas as pd

from model import Model

model_name = "google-bert/bert-base-uncased"
bert_model = Model(model_name)

abdallah_classification = pd.read_csv("../abdallah.csv")
abdallah_classification.dropna(inplace=True)
abdallah_classification = abdallah_classification.rename(columns={'Category': 'category', 'Message': 'sentence'})
labels = ["ham", "spam"]
bert_model.train_evaluate(labels, abdallah_classification)

# Results:
# {'eval_loss': 0.02411392517387867,
# 'eval_accuracy': 0.9940191387559809,
# 'eval_f1': 0.9939839622334854,
# 'eval_runtime': 50.4879,
# 'eval_samples_per_second': 33.117,
# 'eval_steps_per_second': 2.08,
# 'epoch': 2.0}

# dataset url: https://www.kaggle.com/datasets/abdallahwagih/spam-emails