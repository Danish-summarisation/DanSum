from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast

tokenizer = T5TokenizerFast.from_pretrained("/data-big-projects/danish-summarization-danewsroom/models/copper-flower-226/checkpoint-50000")
model = AutoModelForSeq2SeqLM.from_pretrained("/data-big-projects/danish-summarization-danewsroom/models/copper-flower-226/checkpoint-50000")


def generate_summary(batch):
    batch["text"] = ["summarize: " + doc for doc in batch["text"]]
    inputs = tokenizer(
        batch["text"],
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    input_ids = inputs.input_ids.to("cpu")
    attention_mask = inputs.attention_mask.to("cpu")

    # make the model generate predictions (summaries) for articles in text set
    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch
