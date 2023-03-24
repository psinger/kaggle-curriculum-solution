import numpy as np
import pandas as pd
import importlib
import sys
import random
from tqdm import tqdm
import gc
import argparse
import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
from copy import copy
import os
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import SequentialSampler, DataLoader
import yaml
from types import SimpleNamespace  
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append("models")
sys.path.append("datasets")

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_train_dataloader(train_ds, cfg):
    train_dataloader = DataLoader(
        train_ds,
        sampler=None,
        shuffle=True,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.environment.number_of_workers,
        pin_memory=False,
        drop_last=cfg.training.drop_last_batch,
        worker_init_fn=worker_init_fn,
    )
    print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader


def get_scheduler(cfg, optimizer, total_steps):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_epochs * (total_steps // cfg.training.batch_size),
        num_training_steps=cfg.training.epochs * (total_steps // cfg.training.batch_size),
    )
    return scheduler


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_model(cfg):
    Net = importlib.import_module("feedback_metric_model").Net
    return Net(cfg)


parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)


cfg = yaml.safe_load(open(parser_args.config).read())
for k,v in cfg.items():
    if type(v) == dict:
        cfg[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**cfg)
print(cfg)

os.makedirs(f"output/{cfg.experiment_name}", exist_ok=True)

cfg.CustomDataset = importlib.import_module("feedback_metric_ds").CustomDataset

if __name__ == "__main__":

    device = "cuda:0"
    cfg.device = device

    if cfg.environment.seed < 0:
        cfg.environment.seed = np.random.randint(1_000_000)
    else:
        cfg.environment.seed = cfg.environment.seed

    set_seed(cfg.environment.seed)

    train_df = pd.read_csv(cfg.dataset.train_dataframe)

    train_dataset = cfg.CustomDataset(train_df, mode="train", cfg=cfg)

    cfg.train_dataset = train_dataset

    train_dataloader = get_train_dataloader(train_dataset, cfg)

    model = get_model(cfg)
    model.to(device)

    total_steps = len(train_dataset)

    params = model.parameters()

    no_decay = ['bias', 'LayerNorm.weight']
    differential_layers = cfg.training.differential_learning_rate_layers
    optimizer = optim.AdamW(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.learning_rate,
                "weight_decay": cfg.training.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.learning_rate,
                "weight_decay": 0,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.differential_learning_rate,
                "weight_decay": cfg.training.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.differential_learning_rate,
                "weight_decay": 0,
            },
        ],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )


    scheduler = get_scheduler(cfg, optimizer, total_steps)

    if cfg.environment.mixed_precision_training:
        scaler = GradScaler()

    cfg.curr_step = 0
    i = 0
    best_val_loss = np.inf
    optimizer.zero_grad()
    for epoch in range(cfg.training.epochs):

        set_seed(cfg.environment.seed + epoch)

        cfg.curr_epoch = epoch

        print("EPOCH:", epoch)

        progress_bar = tqdm(range(len(train_dataloader)))
        tr_it = iter(train_dataloader)

        losses = []

        gc.collect()

        # ==== TRAIN LOOP
        for itr in progress_bar:
            i += 1

            cfg.curr_step += cfg.training.batch_size

            data = next(tr_it)

            model.train()
            torch.set_grad_enabled(True)

            batch = cfg.CustomDataset.batch_to_device(data, device)

            if cfg.environment.mixed_precision_training:
                with autocast():
                    output_dict = model(batch)
            else:
                output_dict = model(batch)

            loss = output_dict["loss"]

            losses.append(loss.item())

            if cfg.environment.mixed_precision_training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            if cfg.curr_step % cfg.training.batch_size == 0:
                progress_bar.set_description(f"loss: {np.mean(losses[-10:]):.4f}")

        if cfg.training.epochs > 0:
            checkpoint = {
                "model": model.state_dict(),
            }

            torch.save(checkpoint, f"output/{cfg.experiment_name}/checkpoint.pth")
