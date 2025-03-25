# %%
import os
from typing import List

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["WANDB_PROJECT"]="molecular-fingerprinting"

# %%
from datasets import DatasetDict, Value
import torch
from torch import nn
from torch.utils.data import random_split, Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import evaluate
import wandb

# %%
# ds = DatasetDict.load_from_disk('data/dataset')

# ds = ds.rename_column('data', 'inputs')
# ds = ds.remove_columns('sex_label')
# ds = ds.cast_column('label', Value('float64'))

# %%
from src.datasets import LabeledDataset
from src.models import *

# %%
ds = torch.load('data/dataset.pt', weights_only=False)

ds = LabeledDataset(ds, active_label=1)

generator = torch.Generator().manual_seed(42)
splits = random_split(ds, [.8, .1, .1], generator)
ds = {
    'train': splits[0],
    'dev': splits[1],
    'test': splits[2]
}

# %%
model = MLPModel().cuda()

# %%
training_args = TrainingArguments(
    output_dir='models',
    per_device_train_batch_size=1024,
    per_device_eval_batch_size=1024,
    learning_rate=1e-3,
    warmup_ratio=.1,
    num_train_epochs=100,
    weight_decay=.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    # bf16=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=12,
    report_to='wandb'
)

# %%
metrics = evaluate.combine(['accuracy', 'precision', 'recall', 'f1'])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return metrics.compute(logits > 0, labels.astype(bool))

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['dev'],
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(10)
    ]
)

# %%
trainer.train()

# %%
trainer.evaluate(ds['test'], metric_key_prefix='test')
trainer.save_model('models/trained_model')

# %%
wandb.finish()
