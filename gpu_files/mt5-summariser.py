'''
This script contains the code for finetuning a pretrained mT5 model for summarisation on the DaNewsroom dataset.
'''

################################ Importing modules ################################
import nltk
import datasets
from datasets import Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5TokenizerFast
from transformers import EarlyStoppingCallback
import time


################################ Setup stuff ################################
start = time.time()
timestr = time.strftime("%d-%H%M%S")
timestr = timestr + "_sara"
nltk.download('punkt')
#model_checkpoint = "google/mt5-small" # specify model
model_checkpoint = "sarakolding/daT5-base"
metric = datasets.load_metric("rouge")

################################ Load data ################################
# 1k
train = Dataset.from_pandas(pd.read_csv("train50k.csv", usecols=['text','summary'])) # training data
test = Dataset.from_pandas(pd.read_csv("test50k.csv", usecols=['text','summary'])) # test data
val = Dataset.from_pandas(pd.read_csv("val50k.csv", usecols=['text','summary'])) # validation data

#train = Dataset.from_pandas(pd.read_csv("train_abs.csv", usecols=['text','summary'])) # training data
#test = Dataset.from_pandas(pd.read_csv("test_abs.csv", usecols=['text','summary'])) # test data
#val = Dataset.from_pandas(pd.read_csv("val_abs.csv", usecols=['text','summary'])) # validation data

# make into datasetdict format
dd = datasets.DatasetDict({"train":train,"validation":val,"test":test})

################################ Preprocessing ################################
# specify tokenizer
# @ USE DANISH TOKENISER?!

#tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer = T5TokenizerFast.from_pretrained(model_checkpoint)


# add prefix
prefix = "summarize: "
#prefix = ""
#prefix = "opsummér: "

# specify lengths
max_input_length = 1024 # max text (article) max token length
max_target_length = 128 # max reference summary max token length

def preprocess_function(examples):
    # concatenate prefix and article into one input
    inputs = [prefix + doc for doc in examples["text"]]

    # tokenize the input + truncate to max input length
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True) 

    # tokenize the ref summary + truncate to max input length
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True) 
    
    # getting IDs for token
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# make the tokenized datasets using the preprocess function
tokenized_datasets = dd.map(preprocess_function, batched=True)


################################ Fine-tuning ################################

# load the pretrained mT5 model from the Huggingface hub 
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
# set batch size (nr of examples per step)
batch_size = 4
# specify training arguments
args = Seq2SeqTrainingArguments(
    output_dir = "./sara" + timestr,
    evaluation_strategy = "steps",
    save_strategy = "steps",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    #weight_decay=0.01,
    logging_steps=500,  # #(maybe set to 10k???) #(default 500?)
    save_steps=500,  # default = 500
    eval_steps=100,  # (defaults to logging_steps?)
    #warmup_steps=3000,  # set to 3000 lsor full training
    save_total_limit=1,
    num_train_epochs=1,
    predict_with_generate=True,
    overwrite_output_dir= True,
    #fp16=True,
    #fp16_full_eval=True,
    load_best_model_at_end = True,
    metric_for_best_model='loss',
    #WANDB stuff:
    #report_to("wandb")
)

# pad the articles and ref summaries (with -100) to max input length
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred # labels = the reference summaries
    #decode generated summaries from IDs to actual words
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # replace -100 in the labels as we can't decode them, replace with pad token id instead
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #decode reference summaries from IDs to actual words
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    # compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {key: value.mid.fmeasure for key, value in result.items()}
    
    # add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    # round to 4 decimals
    metrics={k: round(v, 4) for k, v in result.items()}

    return metrics

# make the model trainer
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics, 
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# train the model!
trainer.train()

# save results of final validation
result=trainer.evaluate()

from numpy import save
save('./sara' + timestr + '_train', result)  

################################ Testing ################################
model.to('cuda')
# look at test set
test_data = dd['test']

# map data correctly

def generate_summary(batch):
    # prepare test data
    batch['text'] = [prefix + doc for doc in batch["text"]]
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_input_length, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    # make the model generate predictions (summaries) for articles in text set
    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

# generate summaries for test set with the function
results = test_data.map(generate_summary, batched=True, batch_size=batch_size)

pred_str = results["pred"] # the model's generated summary
label_str = results["summary"] # actual ref summary from test set

# compute rouge scores
rouge_output = metric.compute(predictions=pred_str, references=label_str)

# save predictions and rouge scores on test set
from numpy import save
np.save('./sara' + timestr + '_preds.npy', results)
np.save('./sara' + timestr + '_test.npy', rouge_output)

end = time.time()
print("TIME SPENT:")
print(end - start)