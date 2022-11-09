#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:02:53 2022

@author: lesya
"""
import re
import torch 
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"won't", "will not ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text)
    text = ''.join(c for c in text if not c.isnumeric())
    
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # create empty lists to store outputs
    input_ids = []
    attention_masks = []
    
    #for every sentence...
    
    for sent in data:
        # 'encode_plus will':
        # (1) Tokenize the sentence
        # (2) Add the `[CLS]` and `[SEP]` token to the start and end
        # (3) Truncate/Pad sentence to max length
        # (4) Map tokens to their IDs
        # (5) Create attention mask
        # (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text = text_preprocessing(sent),   #preprocess sentence
            add_special_tokens = True,         #Add `[CLS]` and `[SEP]`
            max_length= 177  ,             #Max length to truncate/pad
            pad_to_max_length = True,          #pad sentence to max length 
            return_attention_mask= True        #Return attention mask 
        )
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
        
    #convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    
    return input_ids,attention_masks