<div id="top"></div>
<div align="center">
    
<h2 align="center">Automatic Abstractive Summarisation in Danish</h2>
<h3 align="center">NoDaLiDa 2023 Conference Submission</h2>
  <p align="center">
  <em><a href="https://github.com/idabh"><strong>Ida Bang Hansen</strong></a>, <a href="https://github.com/sarakolding"><strong>Sara Kolding</strong></a>, <a href="https://github.com/katrinenymann"><strong>Katrine Nymann</strong></a></em>, <a href="https://github.com/KennethEnevoldsen"><strong>Kenneth C. Enevoldsen</strong></a> & <strong>Ross Deans Kristensen-McLachlan</strong>
  
  <br />
    <a href="https://huggingface.co/sarakolding/daT5-summariser"><strong>Access our model through huggingface</strong></a>
    <br />
  </p>
</div>

## About The Project

This repository contains the code for creating an automatic abstractive summarisation tool in Danish. We fine-tuned an mT5 model on an abstractive subset of the DaNewsroom dataset.

The model can be used for summarisation of individual news articles using [this notebook](https://github.com/Danish-summarisation/DanSum/blob/main/src/evaluation/generate_summary.ipynb), or through the [huggingface API](https://huggingface.co/sarakolding/daT5-summariser).

### Abstract
Automatic abstractive text summarization is a challenging task in the field of natural language processing. This paper presents a model for domain-specific summarization for Danish news articles. DanSumT5 is an mT5 model fine-tuned on a cleaned subset of the DaNewsroom dataset comprising abstractive article-summary pairs. The resulting state-of-the-art model is evaluated both quantitatively and qualitatively, using ROUGE and BERTScore metrics, along with human rankings of the summaries. We find that although model refinements increase quantitative and qualitative performance, the model is still prone to factual errors. We discuss the limitations of current evaluation methods for automatic abstractive summarization and underline the need for improved metrics and transparency within the field. We suggest that future work should employ techniques for detecting and reducing errors in model output and methods for reference-less evaluation of summaries. <br>
<br>
***Key words:** automatic summarization, transformers, Danish, natural language processing*

### Model performance
These are the quantitative results (mean F1 scores) of our model-generated summaries:

| Metric  | DanSumT5_large |
| ------------- | ------------- |
| **BERTScore**  | 88.97 [88.95, 89.00] |
| **ROUGE-1**  | 23.76 [23.60, 23.91] |
| **ROUGE-2**   | 7.46 [7.35, 7.59]  |
| **ROUGE-L**   | 18.25 [18.12, 18.97]|

### Get started
* The DaNewsroom data set can be accessed upon request (https://github.com/danielvarab/da-newsroom)
* Clone the repo
   ```sh
   git clone https://github.com/Danish-summarisation/DanSum.git
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
<br />
Kenneth C. Enevoldsen - kenneth.enevoldsen@cas.au.dk
<br />
Ross Deans Kristensen-McLachlan - rdkm@cas.au.dk


## Acknowledgments
* Thank you to Daniel Varab for providing us with access to DaNewsroom


