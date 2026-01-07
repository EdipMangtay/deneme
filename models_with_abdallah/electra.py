import pandas as pd

from model import Model

model_name = "google/electra-base-discriminator"
electra_model = Model(model_name)

abdallah_classification = pd.read_csv("../abdallah.csv")
abdallah_classification.dropna(inplace=True)
abdallah_classification = abdallah_classification.rename(columns={'Category': 'category', 'Message': 'sentence'})
labels = ["ham", "spam"]
electra_model.train_evaluate(labels, abdallah_classification)

# Results:
# {'eval_loss': 0.06341511011123657,
# 'eval_accuracy': 0.9880382775119617,
# 'eval_f1': 0.9878697346808037,
# 'eval_runtime': 57.5734,
# 'eval_samples_per_second': 29.041,
# 'eval_steps_per_second': 1.824,
# 'epoch': 2.0}

# dataset url: https://www.kaggle.com/datasets/abdallahwagih/spam-emails