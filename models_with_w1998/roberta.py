import pandas as pd

from model import Model

model_name = "FacebookAI/roberta-base"
roberta_model = Model(model_name)

w1998_classification = pd.read_csv("../_w1998.csv")
w1998_classification.dropna(inplace=True)
w1998_classification = w1998_classification.rename(columns={'spam': 'category', 'text': 'sentence'})
labels = [0, 1]
roberta_model.train_evaluate(labels, w1998_classification)

# Results:
# {'eval_loss': 0.0651375874876976,
# 'eval_accuracy': 0.9860383944153578,
# 'eval_f1': 0.9859722103881567,
# 'eval_runtime': 5.7435,
# 'eval_samples_per_second': 299.297,
# 'eval_steps_per_second': 18.804,
# 'epoch': 2.0}

# dataset url: https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset