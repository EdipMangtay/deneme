import pandas as pd

from model import Model

model_name = "google-bert/bert-base-uncased"
bert_model = Model(model_name)

w1998_classification = pd.read_csv("../_w1998.csv")
w1998_classification.dropna(inplace=True)
w1998_classification = w1998_classification.rename(columns={'spam': 'category', 'text': 'sentence'})
labels = [0, 1]
bert_model.train_evaluate(labels, w1998_classification)

# Results:
# {'eval_loss': 0.02306022308766842,
# 'eval_accuracy': 0.995346131471786,
# 'eval_f1': 0.9953425252211628,
# 'eval_runtime': 49.4435,
# 'eval_samples_per_second': 34.767,
# 'eval_steps_per_second': 2.184,
# 'epoch': 2.0}

# dataset url: https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset