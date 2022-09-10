"""
This script contains the code for finetuning a pretrained mT5 model for summarisation on the DaNewsroom dataset.
"""

# Importing modules
import nltk
import ssl
import numpy as np
import pandas as pd
import wandb
import time
import datasets
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Tokenizer,
)

# Setup
model_checkpoint = "google/mt5-small"
model_name = "mt5-small-25k-baseline"
machine_type = "cuda"
start = time.time()
timestr = time.strftime("%d-%H%M%S")
timestr = timestr + "_" + model_name
nltk.download("punkt")
rouge_metric = datasets.load_metric("rouge")
bert_metric = datasets.load_metric("bertscore")

ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')

wandb.init(project="hyperparameter-search", entity="danish-summarisation")
wandb.run.name = timestr

# Load data
train = Dataset.from_pandas(
    pd.read_csv("/data/danish_summarization_danewsroom/train25k_clean.csv", usecols=["text", "summary"])
)  # training data
test = Dataset.from_pandas(
    pd.read_csv("/data/danish_summarization_danewsroom/test25k_clean.csv", usecols=["text", "summary"])
)  # test data
val = Dataset.from_pandas(
    pd.read_csv("/data/danish_summarization_danewsroom/val25k_clean.csv", usecols=["text", "summary"])
)  # validation data

# make into datasetdict format
dd = datasets.DatasetDict({"train": train, "validation": val, "test": test})

# Preprocessing
# removed fast because of warning message
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)

# specify lengths
max_input_length = 1024  # max text (article) max token length
max_target_length = 128  # max reference summary max token length


def preprocess_function(examples):
    inputs = [doc for doc in examples["text"]]

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

# Fine-tuning
# load the pretrained mT5 model from the Huggingface hub
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_checkpoint,
    min_length=15, 
    max_length=128, 
    num_beams=4,
    no_repeat_ngram_size=3,
    length_penalty=5,
    early_stopping=True,
    dropout_rate=0.01,
)



# specify training arguments
args = Seq2SeqTrainingArguments(
    output_dir="home/sarakolind/" + timestr,
    evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=5e-5,
    lr_scheduler_type="constant",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=100,
    save_steps=200,
    eval_steps=200,
    warmup_steps=100,
    save_total_limit=1,
    num_train_epochs=1,
    predict_with_generate=True,
    overwrite_output_dir=True,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

# pad the articles and ref summaries (with -100) to max input length
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)


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
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# train the model!
trainer.train()

# Testing
model.to(machine_type)
test_data = dd["test"]


def generate_summary(batch):
    # prepare test data
    batch["text"] = [doc for doc in batch["text"]]
    inputs = tokenizer(
        batch["text"],
        padding="max_length",
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
    )
    input_ids = inputs.input_ids.to(machine_type)
    attention_mask = inputs.attention_mask.to(machine_type)

    # make the model generate predictions (summaries) for articles in text set
    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch


# generate summaries for test set with the function
results = test_data.map(generate_summary, batched=True, batch_size=8)

pred_str = results["pred"]  # the model's generated summary
label_str = results["summary"]  # actual ref summary from test set

# compute rouge scores
rouge_output = rouge_metric.compute(predictions=pred_str, references=label_str)
bert_output = bert_metric.compute(predictions=pred_str, references=label_str, lang='da')

# save predictions and rouge scores on test set
results = pd.DataFrame([results])
results.to_csv("home/sarakolind/" + timestr + "_preds.csv")

rouge_output = pd.DataFrame([rouge_output])
rouge_output.to_csv("home/sarakolind/" + timestr + "_rouge.csv")

bert_output = pd.DataFrame([bert_output])
bert_output.to_csv("home/sarakolind/" + timestr + "_bert.csv")

end = time.time()
print("TIME SPENT:")
print(end - start)
