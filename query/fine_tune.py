import torch
from transformers import AutoTokenizer, MobileBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset

model_name = "google/mobilebert-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MobileBertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Adjust `num_labels` for your dataset

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

df = pd.read_csv("/home/aman/code/searchsp/query/data/multimodal_search_dataset.csv")
df = df.rename(columns={"query": "text", "label": "label"})

id2label = {"TEXT" : 0 , "IMAGE" : 1}

for i in range(len(df['label'])):
    df['label'][i] = id2label[df['label'][i]]


hf_dataset = Dataset.from_pandas(df)
tokenized_dataset = hf_dataset.map(preprocess_function , batched=True)

training_args = TrainingArguments(
    output_dir="/home/aman/code/searchsp/query/results",
    evaluation_strategy="no",
    save_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none"
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1).numpy()
    labels = labels.numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=None,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

save_directory = "/home/aman/code/searchsp/query/results/saved_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")