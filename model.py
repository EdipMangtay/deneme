from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from torch.optim import AdamW
import os
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from datasets import Dataset

class Model:
    def __init__(self, model_name):
        self.model_name = model_name

    def train_evaluate(self, spam_labels, df2):
        spam_to_id = {spam: idx for idx, spam in enumerate(spam_labels)}

        os.environ["WANDB_DISABLED"] = "true"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizerE = AutoTokenizer.from_pretrained(self.model_name)

        modelE = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=len(spam_labels), ignore_mismatched_sizes=True).to(device)
        optimizer = AdamW(modelE.parameters(), lr=2e-5, weight_decay=0.1)


        df2["label"] = df2["category"].map(spam_to_id)

        # Converting to Hugging Face Dataset
        dataset = Dataset.from_pandas(df2).shuffle(seed=42)

        def compute_metrics(p):
            preds = p.predictions.argmax(axis=-1)
            labels = p.label_ids
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='weighted')
            return {
                "accuracy": acc,
                "f1": f1,
            }

        # Tokenization Function
        def tokenizer_function(examples):
            return tokenizerE(examples["sentence"], truncation=True, padding=True, max_length=50, add_special_tokens=True)


        tokenized_dataset = dataset.map(tokenizer_function, batched=True)

        # Splitting Dataset into Train and Test Sets
        train_test_split = tokenized_dataset.train_test_split(test_size=0.3)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]

        # Defining Training Arguments
        training_args = TrainingArguments(
            output_dir="./checkpoint",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            save_total_limit=2,
            logging_dir="./logs",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            no_cuda=False,
        )

        # Defining Trainer
        trainer = Trainer(
            model=modelE,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizerE,
            optimizers=(optimizer, None),
            compute_metrics=compute_metrics,
            #callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )


        trainer.train()

        print(trainer.evaluate())

