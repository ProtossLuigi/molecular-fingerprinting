# %%
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["WANDB_PROJECT"]="molecular-fingerprinting"

# %%
from datasets import DatasetDict, Value
from torch import nn
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import evaluate
import wandb

# %%
ds = DatasetDict.load_from_disk('data/dataset')

ds = ds.rename_column('data', 'inputs')
ds = ds.remove_columns('sex_label')
ds = ds.cast_column('label', Value('float64'))

# %%
ds['train'][0]

# %%
class CustomModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        activation_layer = nn.ReLU
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(840 * 4, 1024),
            activation_layer(),
            nn.Linear(1024, 128),
            activation_layer(),
            nn.Linear(128, 1),
            # nn.Sigmoid()
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, labels=None):
        # print(inputs, labels)
        logits = self.net(inputs).squeeze()
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

model = CustomModel().cuda()

# %%
training_args = TrainingArguments(
    output_dir='models',
    per_device_train_batch_size=3200,
    per_device_eval_batch_size=3200,
    learning_rate=1e-3,
    num_train_epochs=100,
    weight_decay=.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    bf16=True,
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
trainer.evaluate(ds['test'])

# %%
wandb.finish()


