"""
This script contains the code for hyperparameter search for a mT5 model for summarisation on the DaNewsroom dataset.
"""

################################ Importing modules ################################
import nltk
import numpy as np
import pandas as pd
import wandb
import time
import datasets
from datasets import Dataset
from ray import tune
# from evaluation import Fragments # does not work
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Tokenizer,
)

################################ Setup ################################
model_checkpoint = "google/mt5-small"
model_name = "mt5-small-25k-baseline"
machine_type = "cuda"
start = time.time()
timestr = time.strftime("%d-%H%M%S")
timestr = timestr + "_" + model_name
nltk.download("punkt")
rouge_metric = datasets.load_metric("rouge")
bert_metric = datasets.load_metric("bertscore")

wandb.init(project="summarisation", entity="idasara")
wandb.run.name = timestr
################################ Load data ################################
# 25 k subset with 89-(10)-1 split
train = Dataset.from_pandas(
    pd.read_csv("train1k.csv", usecols=["text", "summary"])
)  # training data
# test = Dataset.from_pandas(
#     pd.read_csv("test_clean1.csv", usecols=["text", "summary"])
# )  # test data
val = Dataset.from_pandas(
    pd.read_csv("val25k_clean1.csv", usecols=["text", "summary"])
)  # validation data

# make into datasetdict format
dd = datasets.DatasetDict({"train": train, "validation": val}) #, "test": test})

################################ Preprocessing ################################

# removed fast because of warning message
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)

# add prefix
prefix = "summarize: "

# specify lengths
max_input_length = 1024  # max text (article) max token length
max_target_length = 128  # max reference summary max token length


def preprocess_function(examples):
    # concatenate prefix and article into one input
    inputs = [prefix + doc for doc in examples["text"]]

    # tokenize the input + truncate to max input length
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # tokenize the ref summary + truncate to max input length
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"], max_length=max_target_length, truncation=True
        )

    # getting IDs for token
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# make the tokenized datasets using the preprocess function
tokenized_datasets = dd.map(preprocess_function, batched=True)

################################ Hyperparameter search ################################

def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

#wandb.watch(model_init, log_freq=100)

# set batch size (nr of examples per step)
# batch_size = 8
# specify trainiqng arguments
args = Seq2SeqTrainingArguments(
    output_dir="./" + timestr,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_steps=100,
    save_steps=200,
    eval_steps=200,
    warmup_steps=100,
    save_total_limit=1,
    predict_with_generate=True,
    overwrite_output_dir=True,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

# pad the articles and ref summaries (with -100) to max input length
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_init, pad_to_multiple_of=8)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred  # labels = the reference summaries
    # decode generated summaries from IDs to actual words
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # replace -100 in the labels as we can't decode them, replace with pad token id instead
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decode reference summaries from IDs to actual words
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    # compute ROUGE scores
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {key: value.mid.fmeasure for key, value in result.items()}

    # compute BERTScores
    bertscores = bert_metric.compute(predictions=decoded_preds, references=decoded_labels, lang="da")
    result['bertscore'] = np.mean(bertscores['precision'])

    # compute moverscore - not working yet
    # from moverscore_v2 import get_idf_dict, word_mover_score 

    # idf_dict_hyp = get_idf_dict(dec) # idf_dict_hyp = defaultdict(lambda: 1.)
    # idf_dict_ref = get_idf_dict(references) # idf_dict_ref = defaultdict(lambda: 1.)

    # scores = word_mover_score(references, translations, idf_dict_ref, idf_dict_hyp, \
    #                       stop_words=[], n_gram=1, remove_subwords=True)

    # compute density - not working yet
    # fragments = [Fragments(n[0], n[1], lang="da") for n in zip(decoded_preds, texts)]
    # densities = [n.density() for n in fragments]

    # add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    # round to 4 decimals
    metrics = {k: round(v, 4) for k, v in result.items()}

    return metrics


# make the model trainer
trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

def my_hp_space_ray(trial):
    return {
        "learning_rate": tune.linear(5e-4, 5e-6),
        "lr_scheduler_type": tune.categorical('constant', 'linear'), # more
        "num_train_epochs": tune.choice(range(1, 10)),
        # "seed": tune.choice(range(1, 41)), 
        "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
        "num_beams": tune.choice(range(4, 6)), # or without range?
        "no_repeat_ngram_size": tune.choice(range(2, 5)),
        "length_penalty": tune.choice(range(4, 10)),
        "early_stopping": tune.categorical('True'),
        "dropout_rate": tune.linear(0.15, 0.3),
        "max_grad_norm": tune.choice(range(5, 10)) # discrete? no 1?
    }

# dataset density threshold, continuous [1.5-8]
# warm-up steps [10%]

# Default objective is the sum of all metrics
# when metrics are provided, so we have to maximize it.
best_run = trainer.hyperparameter_search(
    direction="maximize", 
    backend="ray", 
    n_trials=10, # number of trials
    # Choose among many libraries:
    # https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
    search_alg=HyperOptSearch(metric="objective", mode="max"),
    # Choose among schedulers:
    # https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
    scheduler=ASHAScheduler(metric="objective", mode="max"),
    hp_space=my_hp_space_ray
)

best_run

# for n, v in best_run.hyperparameters.items():
#     setattr(trainer.args, n, v)

# trainer.train()