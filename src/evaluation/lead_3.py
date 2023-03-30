import nltk
import numpy as np
import wandb

import hydra
from omegaconf import DictConfig
from fragments import Fragments

import datasets
from datasets import load_dataset
from typing import Any, Dict, List, MutableMapping, Tuple, Union

nltk.download("punkt")

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
    example["pred"] = "\n".join(nltk.tokenize.sent_tokenize(example["text"])[:3])
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
    result["bertscore_mean"] = np.mean(bertscores["f1"])
    samples = np.random.choice(bertscores["f1"], 1000)
    percentile_delta = (1 - 0.95) / 2
    q = 100 * np.array([percentile_delta, 0.5, 1 - percentile_delta])
    result["bertscore_low"] = np.percentile(samples, q, axis=0)[0]
    result["bertscore_mid"] = np.percentile(samples, q, axis=0)[1]
    result["bertscore_high"] = np.percentile(samples, q, axis=0)[2]

    # compute density
    fragment = [Fragments(decoded_pred, decoded_input, lang=cfg.language) for decoded_pred, decoded_input in zip(decoded_preds, decoded_inputs)]
    density = [frag.density() for frag in fragment]
    result["density_mean"] = np.mean(density)
    samples = np.random.choice(density, 1000)
    q = 100 * np.array([percentile_delta, 0.5, 1 - percentile_delta])
    result["density_low"] = np.percentile(samples, q, axis=0)[0]
    result["density_mid"] = np.percentile(samples, q, axis=0)[1]
    result["density_high"] = np.percentile(samples, q, axis=0)[2]

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