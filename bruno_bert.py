# Imports
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import classification_report

# 1. Detect device (MPS -> CUDA -> CPU fallback)
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# 2. Load preprocessed data
X = pd.read_csv("X_daily.csv")
y = pd.read_csv("y_daily.csv")
X['target'] = y['target']

# 3. Encode labels
label2id = {'DOWN': 0, 'UP': 1}
id2label = {0: 'DOWN', 1: 'UP'}
X['label'] = X['target'].map(label2id)

# 4. Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['headlines'], truncation=True, padding='max_length', max_length=128)

# 5. Train-test split by unique dates
unique_dates = X['date_used'].unique()
np.random.seed(42)
np.random.shuffle(unique_dates)

split_idx = int(0.8 * len(unique_dates))
train_dates = unique_dates[:split_idx]
test_dates = unique_dates[split_idx:]

train_df = X[X['date_used'].isin(train_dates)].reset_index(drop=True)
test_df = X[X['date_used'].isin(test_dates)].reset_index(drop=True)

print(f"Train samples: {len(train_df)}, Unique train dates: {len(train_df['date_used'].unique())}")
print(f"Test samples: {len(test_df)}, Unique test dates: {len(test_df['date_used'].unique())}")

# 6. Prepare datasets
train_ds = Dataset.from_pandas(train_df[['headlines', 'label']])
test_ds = Dataset.from_pandas(test_df[['headlines', 'label']])

tokenized_train = train_ds.map(tokenize_function, batched=True)
tokenized_test = test_ds.map(tokenize_function, batched=True)

tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 7. Load model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    id2label=id2label,
    label2id=label2id
).to(device)

# 8. Set up training arguments
training_args = TrainingArguments(
    output_dir='./results-date-split',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=0.01,
    logging_steps=10,
    save_strategy='no',
    learning_rate=2e-5,
    seed=42,
    disable_tqdm=True,
    report_to="none"
)

# 9. Set up Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer
)

# Train
trainer.train()

# Move model to CPU
model_cpu = trainer.model.to('cpu')

# Manually predict without Trainer.predict
from torch.utils.data import DataLoader

eval_dataloader = DataLoader(tokenized_test, batch_size=64)

model_cpu.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in eval_dataloader:
        # Only select the needed inputs
        inputs = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        
        outputs = model_cpu(**inputs)
        logits = outputs.logits
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['label'].cpu().numpy())

# Evaluate
from sklearn.metrics import classification_report
print(classification_report(all_labels, all_preds, target_names=['DOWN', 'UP']))
