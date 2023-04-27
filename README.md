<div id="top"></div>
<div align="center">
    <img src="data/logo_DanSumT5.png" alt="Logo" width="70" height="70">
<h2 align="center">DanSumT5: Automatic Abstractive Summarisation in Danish</h3>

  <em><a href="https://github.com/sarakolding"><strong>Sara Kolding</strong></a>, <a href="https://github.com/katrinenymann"><strong>Katrine Nymann</strong></a>, <a href="https://github.com/idabh"><strong>Ida Bang Hansen</strong></a>, <a href="https://github.com/KennethEnevoldsen"><strong>Kenneth C. Enevoldsen</strong></a> & <a href="https://github.com/rdkm89"><strong>Ross Deans Kristensen-McLachlan</strong></a></em>
  <br />
    <a href="https://huggingface.co/Danish-summarisation"><strong>Access our models through huggingface</strong></a>
    <br />
  </p>
</div>

## About The Project

This repository contains the code for creating an automatic abstractive summarisation tool in Danish. We fine-tuned a language-specific pruned mT5 model on an abstractive subset of the DaNewsroom dataset.

The model can be used for summarisation of individual news articles using [this notebook](https://github.com/idabh/data-science-exam/blob/main/generate_summary.ipynb), or through the [huggingface API](https://huggingface.co/sarakolding/daT5-summariser).

### Abstract
Automatic abstractive text summarization
is a challenging task in the field of natural language processing. This paper
presents a model for domain-specific summarization for Danish news articles, DanSumT5; an mT5 model fine-tuned on a
cleaned subset of the DaNewsroom dataset
consisting of abstractive summary-article
pairs. The resulting state-of-the-art model
is evaluated both quantitatively and qualitatively, using ROUGE and BERTScore
metrics and human rankings of the summaries. We find that although model refinements increase quantitative and qualitative performance, the model is still prone
to factual errors. We discuss the limitations of current evaluation methods for automatic abstractive summarization and underline the need for improved metrics and
transparency within the field. We suggest that future work should employ methods for detecting and reducing errors in
model output and methods for referenceless evaluation of summaries. <br>
<br>
***Key words:** automatic summarisation, transformers, Danish, natural language processing*

### Model performance
These are the quantitative results of our model-generated summaries:

| Model |  ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| <a href="https://huggingface.co/Danish-summarisation/DanSum-mT5-small"> **DanSum-mT5-small** </a> | 21.42 [21.26, 21.55]  |  6.21 [6.11, 6.30]  |  16.10 [15.98, 16.22]  | 88.28 [88.26, 88.31] |
| <a href="https://huggingface.co/Danish-summarisation/DanSum-mT5-base"> **DanSum-mT5-base** | 23.21 [23.06, 23.36]  | 7.12 [7.00, 7.22]  | 17.64 [17.50, 17.79]  |  88.77 [88.74, 88.80] |
| <a href="https://huggingface.co/Danish-summarisation/DanSum-mT5-large"> **DanSum-mT5-large** | 23.76 [23.60, 23.91]  |  7.46 [7.35, 7.59]  | 18.25 [18.12, 18.39]  |  88.97 [88.95, 89.00] |

To get a better understanding of the model's performance, we also had two of the authors to blindly (without knowledge of which model generated which summary) rank the model-generated summaries for 100 articles. The results are shown in the table below:


Where reference is the original summary.




### Get started
* The DaNewsroom data set can be accessed upon request (https://github.com/danielvarab/da-newsroom)
* Clone the repo
   ```sh
   git clone https://github.com/idabh/data-science-exam
   ```
* Install required modules
  ```sh
  pip install -r requirements.txt
  ```

## Contact
Ida Bang Hansen - idabanghansen@gmail.com
<br />
Sara Kolding - sarakolding@live.dk
<br />
Katrine Nymann - katrinesofienm@hotmail.dk

## Acknowledgments
*  DAT5 icon created with [OpenAI's DALL-E 2](https://openai.com/dall-e-2/)

