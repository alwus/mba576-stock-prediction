
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
from datasets import Dataset

# Load preprocessed data
from bruno_dataprep import X, y

# Encode labels
label2id = {'DOWN': 0, 'UP': 1}
id2label = {0: 'DOWN', 1: 'UP'}
labels = y.map(label2id)

# Tokenizer and dataset conversion
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True)

dataset = Dataset.from_dict({'text': X.tolist(), 'label': labels.tolist()})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data collator handles dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Model initialization
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# Cross-validation setup
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
predictions = []
true_labels = []

for train_idx, test_idx in kfold.split(X, labels):
    train_split = tokenized_dataset.select(train_idx)
    test_split = tokenized_dataset.select(test_idx)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=50,
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=test_split,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    preds = trainer.predict(test_split)
    preds_labels = preds.predictions.argmax(axis=1)

    predictions.extend(preds_labels)
    true_labels.extend(test_split['label'])

# Report metrics
print(classification_report(true_labels, predictions, target_names=['DOWN', 'UP']))
