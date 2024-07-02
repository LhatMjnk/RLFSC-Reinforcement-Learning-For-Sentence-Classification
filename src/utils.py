import re
import string
import torch
import polars as pl
import pandas as pd
from underthesea import word_tokenize, text_normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import numpy as np

# Load stopwords
stopwords = open("./src/vietnamese-stopwords.txt", encoding="utf-8").readlines()
stopwords = set(word.replace('\n', '') for word in stopwords)

# Load acronyms
acronyms = {
    "ko": "không",
    "k": "không",
    "đc": "được",
    "dc": "được",
    "sp": "sản phẩm",
    "bt": "biết",
    'sd': 'sử dụng',
    'đt': 'điện thoại',
    'trc': 'trước',
    'mn': 'mọi người',
    'r': 'rồi',
}

# Label and rating mapping (using dictionaries)
label_map = {"POS": 0, "NEG": 1, "NEU": 2}
rate_map = {1: 0, 2: 1, 3: 1, 4: 2}

def clean_text(text: str, duplicate=False) -> list[str]:
    # Clean the text
    cleaned_text = text.lower().strip()

    # Remove emojis using regular expressions
    emoji_pattern = re.compile("[" + u"\U0001F600-\U0001F64F" + u"\U0001F300-\U0001F5FF" + 
                               u"\U0001F680-\U0001F6FF" + u"\U0001F1E0-\U0001F1FF" + 
                               u"\U00002702-\U000027B0" + u"\U000024C2-\U0001F251" + 
                               u"\U0001f926-\U0001f937" + u'\U00010000-\U0010ffff' + 
                               u"\u200d" + u"\u2640-\u2642" + u"\u2600-\u2B55" + 
                               u"\u23cf" + u"\u23e9" + u"\u231a" + u"\u3030" + 
                               u"\ufe0f" + "]+", 
                               flags=re.UNICODE)
    cleaned_text = emoji_pattern.sub(r'', cleaned_text)

    # Remove punctuation using str.translate
    cleaned_text = cleaned_text.translate(str.maketrans("", "", string.punctuation))

    # Normalize text
    cleaned_text = text_normalize(cleaned_text)

    # Remove extra whitespace
    cleaned_text = re.sub(' +', ' ', cleaned_text)

    tokens = word_tokenize(cleaned_text)
    tokens = [acronyms.get(token, token) for token in tokens]
    cleaned_text_with_stopword = " ".join(tokens)

    if duplicate:
        # Remove Vietnamese stopwords
        tokens = [token for token in tokens if token not in stopwords]
        cleaned_text_without_stopword = " ".join(tokens)

        # Duplicate the cleaned text and label with a list comprehension
        cleaned_texts = [cleaned_text_with_stopword, cleaned_text_without_stopword]
    else:
        cleaned_texts = cleaned_text_with_stopword

    return cleaned_texts

def map_label(label):
    return label_map.get(label)

def label_decode(label):
    if label == 0:
        return 'Negative'
    elif label == 1:
        return 'Neutral'
    elif label == 2:
        return 'Positive'

def map_rating(rate):
    return rate_map.get(rate)

def preprocessing_data(data: pl.DataFrame, name:str=None) -> pl.DataFrame:
    # Apply the clean_text function to each row
    cleaned_texts = data["content"].apply(clean_text, duplicate=True)

    # Duplicate the labels
    labels = data["rating"].apply(lambda x: [x, x])

    # Flatten the lists
    all_cleaned_texts = [text for sublist in cleaned_texts for text in sublist]
    all_labels = [label for sublist in labels for label in sublist] # duplicate the labels

    # Create the new DataFrame with two columns
    new_data = pl.DataFrame({"content": all_cleaned_texts, "rating": all_labels})
    new_data = new_data.with_columns(pl.col("rating").map_elements(map_rating, return_dtype=pl.Int8).alias("rating"))

    new_data = new_data.unique("content")
    new_data = new_data.filter(pl.col("content").map_elements(len, return_dtype=pl.Int32) > 2)
    new_data = new_data.drop_nulls()

    new_data = new_data.with_columns(pl.col("content").map_elements(lambda x: word_tokenize(x, format="text"), return_dtype=pl.Utf8).alias("content"))

    if name:
        new_data.write_csv(f"./Data/Processed/{name}.csv")

    return new_data

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def prepare_dataloaders(data:pd.DataFrame, tokenizer, batch_size, max_len):
    # Vectorization using count vectorizer
    encode_data = tokenizer.batch_encode_plus(data['content'].values.tolist(), 
                                              max_length=max_len, 
                                              padding='max_length', 
                                              truncation=True, 
                                              return_tensors='pt',
                                              return_attention_mask=True)

    labels = torch.tensor(data['rating'].values)
    input_ids = encode_data['input_ids']
    attention_masks = encode_data['attention_mask']

    X_train, X_val, y_train, y_val, X_train_masks, X_val_masks = train_test_split(input_ids, labels, attention_masks, test_size=0.05, random_state=42)
    X_val, X_test, y_val, y_test, X_val_masks, X_test_masks = train_test_split(X_val, y_val, X_val_masks, test_size=0.5, random_state=42)

    train_data = TensorDataset(X_train, X_train_masks, y_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(X_val, X_val_masks, y_val)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    test_data = TensorDataset(X_test, X_test_masks, y_test)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader