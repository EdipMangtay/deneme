import pandas as pd

from model import Model

model_name = "YituTech/conv-bert-base"
convbert_model = Model(model_name)

w1998_classification = pd.read_csv("../_w1998.csv")
w1998_classification.dropna(inplace=True)
w1998_classification = w1998_classification.rename(columns={'spam': 'category', 'text': 'sentence'})
labels = [0, 1]
convbert_model.train_evaluate(labels, w1998_classification)

# Results:
# {'eval_loss': 0.05534691363573074,
# 'eval_accuracy': 0.987783595113438,
# 'eval_f1': 0.987750139832619,
# 'eval_runtime': 7.526,
# 'eval_samples_per_second': 228.407,
# 'eval_steps_per_second': 14.35,
# 'epoch': 2.0}

# dataset url: https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset