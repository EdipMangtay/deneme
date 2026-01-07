import pandas as pd

from model import Model

model_name = "google/electra-base-discriminator"
electra_model = Model(model_name)

w1998_classification = pd.read_csv("../_w1998.csv")
w1998_classification.dropna(inplace=True)
w1998_classification = w1998_classification.rename(columns={'spam': 'category', 'text': 'sentence'})
labels = [0, 1]
electra_model.train_evaluate(labels, w1998_classification)

# Results:
# {'eval_loss': 0.05735341086983681,
# 'eval_accuracy': 0.987783595113438,
# 'eval_f1': 0.9877107835529795,
# 'eval_runtime': 5.8566,
# 'eval_samples_per_second': 293.514,
# 'eval_steps_per_second': 18.441,
# 'epoch': 2.0}

# dataset url: https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset