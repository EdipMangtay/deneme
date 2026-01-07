import pandas as pd

from model import Model

model_name = "google-bert/bert-base-uncased"
bert_model = Model(model_name)

kucev_classification = pd.read_csv("../kucev.csv")
kucev_classification.drop(columns=["title"], inplace=True)
kucev_classification.dropna(inplace=True)
kucev_classification = kucev_classification.rename(columns={'type': 'category', 'text': 'sentence'})
labels = ["not spam", "spam"]
bert_model.train_evaluate(labels, kucev_classification)

# Results:
# {'eval_loss': 0.5377640128135681,
# 'eval_accuracy': 0.7692307692307693,
# 'eval_f1': 0.6688963210702341,
# 'eval_runtime': 0.9173,
# 'eval_samples_per_second': 28.345,
# 'eval_steps_per_second': 2.18,
# 'epoch': 2.0}

# dataset url: https://www.kaggle.com/datasets/tapakah68/email-spam-classification