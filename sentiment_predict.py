import tensorflow as tf
from transformers import InputExample, InputFeatures
from transformers import BertTokenizer, TFBertForSequenceClassification
from nltk.corpus import stopwords
import string
import re
import nltk
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np

from nltk.tokenize import TweetTokenizer

import sys
import os





nltk.download('stopwords')
stopwords_english = stopwords.words('english')

def process_text(text, get_tokens=False):
    global tokenizer
    global stopwords_english

    # it will remove single numeric terms in the tweet.
    text2 = re.sub(r'[0-9]', '', text2)
    return text2

def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN): 
  train_InputExamples = train.apply(lambda x: InputExample(guid=None, text_a = x[DATA_COLUMN], text_b = None, label = x[LABEL_COLUMN]), axis = 1)

  validation_InputExamples = test.apply(lambda x: InputExample(guid=None, text_a = x[DATA_COLUMN], text_b = None, label = x[LABEL_COLUMN]), axis = 1)
  
  return train_InputExamples, validation_InputExamples


  
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )

def clear_screen():
    os.system('cls')

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_model():
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    clear_screen()


    model.load_weights('./model/v1_model_weights')
    clear_screen()


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])
    
    return model 

def predict_sentiment(text, model):
    text2 = process_text(text)
    samp_example = InputExample(guid=None,text_a = text2, text_b = None,label = 0)
    validation_sample = convert_examples_to_tf_dataset([samp_example], tokenizer).batch(32)
    ans = model.predict(validation_sample)
    ans_pred = np.argmax(ans[0], axis=1)
    clear_screen()
    if ans_pred[0] == 0:
        return 'Negative'
    elif ans_pred[0] == 1:
        return 'Neutral'
    else:
        return 'Positive'

