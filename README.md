<div id="top"></div>

<h3 align="center">Automatic Abstractive Summarisation in Danish</h3>

  <p align="center">
    Data Science Exam - MSc Cognitive Science at Aarhus University - Spring 2022
  <br />
  <em>Ida Bang Hansen & Sara Kolding</em>
  <br />
    <a href="https://huggingface.co/sarakolding/daT5-summariser"><strong>Access our model through huggingface</strong></a>
    <br />
  </p>
</div>

## About The Project

This repository contains the code for creating an automatic abstractive summarisation tool in Danish. We fine-tuned an target language-specific mT5 model on an abstractive subset of the DaNewsroom dataset.

The model can be used for summarisation of individual news articles using [this notebook](https://github.com/idabh/data-science-exam/blob/main/generate_summary.ipynb), or through the [huggingface API](https://huggingface.co/sarakolding/daT5-summariser).

### Abstract
*Automatic abstractive text summarisation is a challenging task in the field of natural language processing. This paper aims to further develop and refine previous work by the authors in domain-specific automatic summarisation for Danish news articles. We extend that work by cleaning the data, pruning the vocabulary multilingual model, and improving the parameter tuning and model selection, as well as evaluating results using additional metrics.
We fine-tune a pruned mT5 model on a cleaned subset of the DaNewsroom dataset consisting of abstractive summary-article pairs. The resulting model is evaluated quantitatively using ROUGE, BERTScore and density measures, and qualitatively by comparing the generated summaries to our previous work. We find that though model refinements increase quantitative and qualitative performance, the model is prone to hallucinations, and the resulting ROUGE scores are in the lower range of comparable abstractive summarisation efforts in other languages. A discussion of the limitations of the current evaluation methods for automatic abstractive summarisation underline the need for improved metrics and transparency within the field. Future work could employ methods for detecting and reducing hallucinations in model output, and employ methods for reference-less evaluation of summaries.*

### Get started
* The DaNewsroom data set can be accessed upon request (https://github.com/danielvarab/da-newsroom)
* Install required modules
  ```sh
  pip install -r requirements.txt
  ```
* Clone the repo
   ```sh
   git clone https://github.com/idabh/data-science-exam
   ```

## Contact
Ida Bang Hansen - idabanghansen@gmail.com
<br />
Sara Kolding - sarakolding@live.dk
