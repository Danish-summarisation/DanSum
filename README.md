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

The model can be used for summarisation of individual, novel news articles using generate_summary.ipynb, or through the [huggingface API](https://huggingface.co/sarakolding/daT5-summariser).

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
