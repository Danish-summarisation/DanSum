from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast

tokenizer = T5TokenizerFast.from_pretrained(
    "/data-big-projects/danish-summarization-danewsroom/models/earnest-shadow-289/checkpoint-129070"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "/data-big-projects/danish-summarization-danewsroom/models/earnest-shadow-289/checkpoint-129070"
)


def generate_summary(batch):

    # prepare test data
    # batch["text"] = [doc for doc in batch["text"]]
    inputs = tokenizer(
        batch,
        padding="max_length",
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )

    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    # make the model generate predictions (summaries) for articles in text set
    outputs = model.to("cuda").generate(input_ids, attention_mask=attention_mask)

    # all special tokens will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return output_str
