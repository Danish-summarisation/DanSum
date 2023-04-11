"""
This script contains the code for finetuning a pretrained mT5 model for summarisation
on the DaNewsroom dataset.

This code can be run using

```bash
python train.py
```

if you want to overwrite specific parameters, you can do so by passing them as arguments to the script, e.g.

```bash
python train.py --config config.yaml training_data.max_input_length=512
```

or by passing a config file with the `--config` flag, e.g.

```bash
python train.py --config config.yaml
```
"""
import os
import time
import ssl
from functools import partial

import nltk

import numpy as np
import pandas as pd

import wandb

import hydra
from omegaconf import DictConfig

import datasets
from fragments import Fragments

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Tokenizer,
    AutoTokenizer,
)

from utils import flatten_nested_config


def preprocess_function(examples, tokenizer, cfg):
    cfg = cfg.training_data
    inputs = [doc for doc in examples["text"]]

    # tokenize the input + truncate to max input length
    model_inputs = tokenizer(inputs, max_length=cfg.max_input_length, truncation=True)

    # tokenize the ref summary + truncate to max input length

    labels = tokenizer(
        examples["summary"], max_length=cfg.max_target_length, truncation=True
    )

    # getting IDs for token
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def generate_summary(batch, tokenizer, model, cfg):
    # prepare test data
    batch["text"] = [doc for doc in batch["text"]]
    inputs = tokenizer(
        batch["text"],
        padding="max_length",
        return_tensors="pt",
        max_length=cfg.training_data.max_input_length,
        truncation=True,
    )
    input_ids = inputs.input_ids.to(cfg.device)
    attention_mask = inputs.attention_mask.to(cfg.device)

    # make the model generate predictions (summaries) for articles in text set
    outputs = model.to(cfg.device).generate(input_ids, attention_mask=attention_mask)

    # all special tokens will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch


def compute_metrics(eval_pred, tokenizer, cfg):

    rouge_metric = datasets.load_metric("rouge")
    bert_metric = datasets.load_metric("bertscore")

    predictions, labels, inputs = eval_pred  # labels = the reference summaries
    # decode generated summaries from IDs to actual words
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # replace -100 in the labels as we can't decode them, replace with pad token id instead
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decode reference summaries from IDs to actual words
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # replace -100 in the labels as we can't decode them, replace with pad token id instead
    inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
    # decode articles from IDs to actual words
    decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]
    decoded_inputs = [
        "\n".join(nltk.sent_tokenize(input.strip())) for input in decoded_inputs
    ]

    # compute ROUGE scores
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {key: value.mid.fmeasure for key, value in result.items()}

    # compute BERTScores
    bertscores = bert_metric.compute(
        predictions=decoded_preds, references=decoded_labels, lang=cfg.language
    )
    result["bertscore"] = np.mean(bertscores["f1"])

    # compute BERTScores
    bertscores_r = bert_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        lang=cfg.language,
        model_type="xlm-roberta-large",
    )
    result["bertscore_r"] = np.mean(bertscores_r["f1"])

    # compute density
    fragment = [
        Fragments(decoded_pred, decoded_input, lang=cfg.language)
        for decoded_pred, decoded_input in zip(decoded_preds, decoded_inputs)
    ]
    density = [frag.density() for frag in fragment]
    result["density"] = np.mean(density)

    # add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    # round to 4 decimals
    metrics = {k: round(v, 4) for k, v in result.items()}

    # log predictions on wandb
    artifact = wandb.Artifact("summaries-" + str(wandb.run.name), type="predictions")
    summary_table = wandb.Table(
        columns=["references", "predictions"],
        data=[
            [ref, pred]
            for ref, pred in zip(decoded_labels[0:100], decoded_preds[0:100])
        ],
    )
    artifact.add(summary_table, "summaries")
    wandb.run.log_artifact(artifact)

    return metrics


def setup_nltk():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download("punkt")


@hydra.main(
    config_path="../../configs", config_name="default_config", version_base="1.2"
)
def main(cfg: DictConfig) -> None:
    """
    Main function for training the model.
    """
    # Setup
    # Setting up wandb
    run = wandb.init(
        project=cfg.project_name,
        config=flatten_nested_config(cfg),
        mode=cfg.wandb_mode,
        entity=cfg.wandb_entity,
    )

    setup_nltk()

    # load dataset
    # dataset = load_dataset(cfg)
    dataset = load_dataset(
        "csv",
        data_files={
            "train": cfg.training_data.train_path,
            "validation": cfg.training_data.val_path,
        },
        cache_dir=cfg.cache_dir,
    )
    start = time.time()

    # Preprocessing
    # removed fast because of warning message
    tokenizer = T5Tokenizer.from_pretrained(cfg.model_checkpoint)

    # make the tokenized datasets using the preprocess function
    tokenized_datasets = dataset.map(
        lambda batch: preprocess_function(batch, tokenizer, cfg),
        batched=True,
        load_from_cache_file=not cfg.redo_cache,
        cache_file_names={
            "train": os.path.join(cfg.cache_dir, "train"),
            "validation": os.path.join(cfg.cache_dir, "val"),
        },
    )

    if cfg.training_data.quality_filter:
        tokenized_datasets = tokenized_datasets.filter(lambda x: x["passed"] is True)
    summary_types = cfg.training_data.summary_type  # a list

    if "mixed" not in summary_types:
        tokenized_datasets["train"] = tokenized_datasets["train"].filter(
            lambda x: x["density_bin"] != "mixed"
        )

    if "extractive" not in summary_types:
        tokenized_datasets["train"] = tokenized_datasets["train"].filter(
            lambda x: x["density_bin"] != "extractive"
        )

    # Fine-tuning
    # load the pretrained mT5 model from the Huggingface hub
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model_checkpoint,
        min_length=cfg.model.min_length,
        max_length=cfg.model.max_length,
        num_beams=cfg.model.num_beams,
        no_repeat_ngram_size=cfg.model.no_repeat_ngram_size,
        length_penalty=cfg.model.length_penalty,
        early_stopping=cfg.model.early_stopping,
        dropout_rate=cfg.model.dropout_rate,
    )

    # specify training arguments
    args = Seq2SeqTrainingArguments(
        output_dir=cfg.training.output_dir + run.name,
        evaluation_strategy=cfg.training.evaluation_strategy,
        save_strategy=cfg.training.save_strategy,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        logging_steps=cfg.training.logging_steps,
        # save_steps=cfg.training.save_steps,
        # eval_steps=cfg.training.eval_steps,
        warmup_steps=cfg.training.warmup_steps,
        save_total_limit=cfg.training.save_total_limit,
        num_train_epochs=cfg.training.num_train_epochs,
        predict_with_generate=cfg.training.predict_with_generate,
        overwrite_output_dir=cfg.training.overwrite_output_dir,
        fp16=cfg.training.fp16,
        # load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        max_grad_norm=cfg.training.max_grad_norm,
        include_inputs_for_metrics=cfg.training.include_inputs_for_metrics,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        # max_steps=cfg.training.max_steps
    )

    # pad the articles and ref summaries (with -100) to max input length
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, pad_to_multiple_of=cfg.training.pad_to_multiple_of
    )

    # make the model trainer
    _compute_metrics = partial(compute_metrics, tokenizer=tokenizer, cfg=cfg)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        # limit eval dataset
        # eval_dataset = tokenized_datasets["validation"].select(range(cfg.training_data.max_eval_samples)),
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    # train the model!
    trainer.train()

    end = time.time()
    print("TIME SPENT:")
    print(end - start)

    eval_score = trainer.evaluate()
    run.finish()
    return eval_score["eval_loss"]


if __name__ == "__main__":
    main()
