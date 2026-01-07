import pandas as pd

from model import Model

model_name = "YituTech/conv-bert-base"
convbert_model = Model(model_name)

abdallah_classification = pd.read_csv("../abdallah.csv")
abdallah_classification.dropna(inplace=True)
abdallah_classification = abdallah_classification.rename(columns={'Category': 'category', 'Message': 'sentence'})
labels = ["ham", "spam"]
convbert_model.train_evaluate(labels, abdallah_classification)

# Results:
# {'eval_loss': 0.03928332403302193,
# 'eval_accuracy': 0.9922248803827751,
# 'eval_f1': 0.9922048104831724,
# 'eval_runtime': 8.9422,
# 'eval_samples_per_second': 186.978,
# 'eval_steps_per_second': 11.742,
# 'epoch': 2.0}

# dataset url: https://www.kaggle.com/datasets/abdallahwagih/spam-emails