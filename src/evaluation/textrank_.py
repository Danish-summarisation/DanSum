#add packages
from math import sqrt
from operator import itemgetter

import spacy
import pytextrank

import pandas as pd
import numpy as np
import wandb

import hydra
from omegaconf import DictConfig
from fragments import Fragments

import datasets
from datasets import load_dataset
from typing import Any, Dict, List, MutableMapping, Tuple, Union

# load spacy model
# load a spaCy model, depending on language, scale, etc.
nlp = spacy.load("en_core_web_sm")
# add PyTextRank to the spaCy pipeline
nlp.add_pipe("textrank")

def flatten_nested_config(
    config: Union[Dict, MutableMapping],
    parent_key: str = "",
    sep: str = ".",
) -> Dict:
    """Recursively flatten an infinitely nested config. E.g. {"level1":

    {"level2": "level3": {"level4": 5}}}} becomes:

    {"level1.level2.level3.level4": 5}.

    Args:
        d (Union[Dict, MutableMapping]): Dict to flatten.
        parent_key (str): The parent key for the current dict, e.g. "level1" for the
            first iteration. Defaults to "".
        sep (str): How to separate each level in the dict. Defaults to ".".

    Returns:
        Dict: The flattened dict.
    """

    items: List[Tuple[str, Any]] = []
    for k, v in config.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(
                flatten_nested_config(config=v, parent_key=new_key, sep=sep).items()
            )
        else:
            items.append((new_key, v))
    return dict(items)


def three_sentence_summary(example):
    # get spaCy doc
    doc = nlp(example["text"])

    # examine the top-ranked phrases in the document
    sent_bounds = [[s.start, s.end, set([])] for s in doc.sents ]

    # limit to 4 phrases
    limit_phrases = 4
    phrase_id = 0
    unit_vector = []
    # get vector for chunks
    for p in doc._.phrases:
        unit_vector.append(p.rank)
        for chunk in p.chunks:
            for sent_start, sent_end, sent_vector in sent_bounds:
                if chunk.start >= sent_start and chunk.end <= sent_end:
                    sent_vector.add(phrase_id)
                    break
        phrase_id += 1
        if phrase_id == limit_phrases:
            break

    # sum the ranks
    sum_ranks = sum(unit_vector)
    unit_vector = [ rank/sum_ranks for rank in unit_vector ]

    # rank sentences by vectors
    sent_rank = {}
    sent_id = 0
    for sent_start, sent_end, sent_vector in sent_bounds:
        sum_sq = 0.0
        for phrase_id in range(len(unit_vector)):
            if phrase_id not in sent_vector:
                sum_sq += unit_vector[phrase_id]**2.0
        sent_rank[sent_id] = sqrt(sum_sq)
        sent_id += 1

    # find 5 most highly ranked sentences
    limit_sentences = 3
    sent_text = {}
    sent_id = 0
    for sent in doc.sents:
        sent_text[sent_id] = sent.text
        sent_id += 1

    # get outputs
    output = []
    num_sent = 0
    for sent_id, rank in sorted(sent_rank.items(), key=itemgetter(1)):
        output.append((sent_id, sent_text[sent_id]))
        num_sent += 1
        if num_sent == limit_sentences:
            break

    # order outputs and print
    pred = []
    for result in sorted(output):
        pred.append(result[1].strip())
    
    # save result and update counter
    example["pred"] = ' '.join(pred)
    
    return example


def compute_metrics(decoded_preds, decoded_labels, decoded_inputs, cfg):

    rouge_metric = datasets.load_metric("rouge")
    bert_metric = datasets.load_metric("bertscore")

    # compute ROUGE scores
    rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_aggregator=True)
    mid = {f"{key}_mid": value.mid.fmeasure for key, value in rouge.items()}
    low = {f"{key}_low": value.low.fmeasure for key, value in rouge.items()}
    high = {f"{key}_high": value.high.fmeasure for key, value in rouge.items()}
    result = {"low": low, "mid": mid, "high": high}

    # # compute BERTScores
    # bertscores = bert_metric.compute(
    #     predictions=decoded_preds, references=decoded_labels, lang=cfg.language
    # )
    # result["bertscore"] = np.mean(bertscores["f1"])

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

    if cfg.training_data.quality_filter:
        dataset = dataset.filter(lambda x: x["passed"] is True)
    summary_types = cfg.training_data.summary_type  # a list

    results = dataset["test"]
    results = results.map(three_sentence_summary)
    
    metrics = compute_metrics(results["pred"], results["summary"], results["text"], cfg)
    wandb.log({"eval": metrics})

    run.finish()

    return metrics

if __name__ == "__main__":
    main()
