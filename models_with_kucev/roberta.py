import pandas as pd

from model import Model

model_name = "FacebookAI/roberta-base"
roberta_model = Model(model_name)

kucev_classification = pd.read_csv("../kucev.csv")
kucev_classification.drop(columns=["title"], inplace=True)
kucev_classification.dropna(inplace=True)
kucev_classification = kucev_classification.rename(columns={'type': 'category', 'text': 'sentence'})
labels = ["not spam", "spam"]
roberta_model.train_evaluate(labels, kucev_classification)

# Results:
# {'eval_loss': 0.6810759902000427,
# 'eval_accuracy': 0.7692307692307693,
# 'eval_f1': 0.7220279720279721,
# 'eval_runtime': 0.7416,
# 'eval_samples_per_second': 35.058,
# 'eval_steps_per_second': 2.697,
# 'epoch': 2.0}

# dataset url: https://www.kaggle.com/datasets/tapakah68/email-spam-classification
