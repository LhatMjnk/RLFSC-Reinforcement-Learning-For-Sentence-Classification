# README.md

## RLFSC (Reinforcement Learning For Sentence Classification)

This project demonstrates the application of Deep Reinforcement Learning (DRL) for sentiment analysis on the Vietnamese Sentiment Classification dataset. It utilizes the PhoBERT model, a pre-trained language model specifically designed for the Vietnamese language, to understand and classify sentiments in Vietnamese texts.

### Project Structure

- [`Data/`]: Contains the UIT-VSFC dataset divided into training, validation, and test sets.
- [`Images/`]: Directory for storing images, such as plots or diagrams, generated during analysis or model training.
- [`Model/`]: Contains saved models and checkpoints.
- [`Notebook/`]: Jupyter notebooks are used for model development, training, and evaluation.
- [`Report/`]: Documentation and reports related to the project.
- [`src/`]: Source code including utility functions, model definitions, and environment configurations.
- [`requirement.txt`]: Lists all the dependencies required to run the project.

### Setup

1. Clone the repository to your local machine.
2. Ensure you have Python 3.6+ installed.
3. Install the required dependencies by running:
   ```shell
   pip install -r requirement.txt
   ```
4. Navigate to the project directory and start Jupyter Notebook:
   ```shell
   jupyter notebook
   ```
5. Open the notebooks to start exploring the project.

### Usage

The notebooks guide you through the process of loading the data, preprocessing, model training, and evaluation. Follow the steps in the notebook for a detailed walkthrough.

### Data

The default dataset is structured as follows:
- `Processed/`: Preprocessed data.
- `Raw/`: Raw collected data.

The UIT-VSFC dataset is structured as follows:
- `train/`: Training data.
- `dev/`: Validation data.
- `test/`: Test data.

Each directory contains:
- `sentiments.txt`: Sentiment labels for each sentence.
- `sents.txt`: The sentences.

### Model

This project uses the `vinai/phobert-base-v2` model from Hugging Face's Transformers library as the feature extraction for sentiment classification.

### Acknowledgments
- The default data belongs to Phenikaa University.
- The UIT datasets are provided by the University of Information Technology, VNU-HCM.
- The PhoBERT model is developed by VinAI Research.
