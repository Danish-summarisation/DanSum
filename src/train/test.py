import os

import nltk

import numpy as np

import wandb

import hydra
from omegaconf import DictConfig

import datasets
from fragments import Fragments

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoTokenizer,
)

from utils import flatten_nested_config

from train import (preprocess_function, generate_summary)

import torch

def compute_metrics(predictions, labels, inputs, tokenizer, cfg):

    rouge_metric = datasets.load_metric("rouge")
    bert_metric = datasets.load_metric("bertscore")

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in labels
    ]
    decoded_inputs = [
        "\n".join(nltk.sent_tokenize(input.strip())) for input in inputs
    ]

    # compute ROUGE scores
    rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_aggregator=True)
    mid = {f"{key}_mid": value.mid.fmeasure for key, value in rouge.items()}
    low = {f"{key}_low": value.low.fmeasure for key, value in rouge.items()}
    high = {f"{key}_high": value.high.fmeasure for key, value in rouge.items()}
    result = {"low": low, "mid": mid, "high": high}

    # compute BERTScores
    bertscores = bert_metric.compute(
        predictions=decoded_preds, references=decoded_labels, lang=cfg.language, model_type="xlm-roberta-large"
    )
    mean_bertscore = [None] * 1000

    for i in range(1000):
      sample_idx = np.random.choice(
          np.arange(len(bertscores["f1"])), size=len(bertscores["f1"]))
      sample = [bertscores["f1"][x] for x in sample_idx]
      mean_bertscore[i] = np.mean(sample)
    
    percentile_delta = (1 - 0.95) / 2
    q = 100 * np.array([percentile_delta, 0.5, 1 - percentile_delta])
    result["bertscore_low"] = np.percentile(mean_bertscore, q)[0]
    result["bertscore_mid"] = np.percentile(mean_bertscore, q)[1]
    result["bertscore_high"] = np.percentile(mean_bertscore, q)[2]
    result["bertscore_mean"] = np.mean(bertscores["f1"])

    # compute density
    fragment = [Fragments(decoded_pred, decoded_input, lang=cfg.language) for decoded_pred, decoded_input in zip(decoded_preds, decoded_inputs)]
    density = [frag.density() for frag in fragment]
    
    mean_density = [None] * 1000

    for i in range(1000):
      sample_idx = np.random.choice(
          np.arange(len(density)), size=len(density))
      sample = [density[x] for x in sample_idx]
      mean_density[i] = np.mean(sample)
    
    percentile_delta = (1 - 0.95) / 2
    q = 100 * np.array([percentile_delta, 0.5, 1 - percentile_delta])
    result["density_low"] = np.percentile(mean_density, q)[0]
    result["density_mid"] = np.percentile(mean_density, q)[1]
    result["density_high"] = np.percentile(mean_density, q)[2]
    result["density_mean"] = np.mean(density)


    metrics = result

    # log predictions on wandb
    artifact = wandb.Artifact("summaries-" + str(wandb.run.name), type="predictions")
    summary_table = wandb.Table(columns=['references', 'predictions'], data=[[ref, pred] for ref, pred in zip(decoded_labels[0:100], decoded_preds[0:100])])
    artifact.add(summary_table, "summaries")
    wandb.run.log_artifact(artifact)      

    return metrics

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

    dataset = load_dataset(
            "csv",
            data_files={
                #"train": cfg.training_data.train_path,
                #"validation": cfg.training_data.val_path,
                "test": cfg.training_data.test_path
            },
            cache_dir=cfg.cache_dir,
        )

    # Preprocessing
    # removed fast because of warning message
    tokenizer = T5Tokenizer.from_pretrained(cfg.model_checkpoint)

    # make the tokenized datasets using the preprocess function
    tokenized_datasets = dataset.map(
        lambda batch: preprocess_function(batch, tokenizer, cfg),
        batched=True,
        load_from_cache_file=not cfg.redo_cache,
        cache_file_names={
            #"train": os.path.join(cfg.cache_dir, "train"),
            #"validation": os.path.join(cfg.cache_dir, "val"),
            "test": os.path.join(cfg.cache_dir, "test")
        },
    )

    if cfg.training_data.quality_filter:
        tokenized_datasets = tokenized_datasets.filter(lambda x: x["passed"] is True)
    summary_types = cfg.training_data.summary_type  # a list

    # subset - delete later!
    # tokenized_datasets['test'] = tokenized_datasets['test'].select(range(cfg.training_data.max_eval_samples))

    tokenizer = T5Tokenizer.from_pretrained(cfg.model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_checkpoint, num_beams=cfg.model.num_beams) #, num_beam_groups=2, diversity_penalty=2)

    results = tokenized_datasets['test'].map(lambda batch: generate_summary(batch, tokenizer, model, cfg), 
        batched=True, 
        batch_size=32 # cfg.eval_batch_size?
        )
    
    metrics = compute_metrics(results['pred'], results['summary'], results['text'], tokenizer, cfg)
    wandb.log({'eval': metrics})

    run.finish()

    return metrics

if __name__ == "__main__":
    main()