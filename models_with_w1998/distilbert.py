import pandas as pd

from model import Model

model_name = "distilbert/distilbert-base-uncased"
distilbert_model = Model(model_name)

w1998_classification = pd.read_csv("../_w1998.csv")
w1998_classification.dropna(inplace=True)
w1998_classification = w1998_classification.rename(columns={'spam': 'category', 'text': 'sentence'})
labels = [0, 1]
distilbert_model.train_evaluate(labels, w1998_classification)

# Results:
# {'eval_loss': 0.05365796387195587,
# 'eval_accuracy': 0.9866201279813845,
# 'eval_f1': 0.9865620998773673,
# 'eval_runtime': 3.0499,
# 'eval_samples_per_second': 563.621,
# 'eval_steps_per_second': 35.411,
# 'epoch': 2.0}

# dataset url: https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset