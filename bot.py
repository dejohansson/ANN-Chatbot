#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
import numpy as np
from matplotlib import pyplot
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from transformers import *
import time
import os
import pandas as pd
import codecs
import re

class BertTextProcessor():
    def __init__(self, tokenizer, max_length = 100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id

        self.create_token = lambda x: [self.tokenizer.tokenize(i)[:self.max_length-2] for i in x]
        self.token2id = lambda x: [self.tokenizer.convert_tokens_to_ids(i) for i in x]
        self.id2string = lambda x: self.tokenizer.convert_tokens_to_string([self.tokenizer.convert_ids_to_tokens(i) for i in x])
        
    def processString(self, text):
      o = self.preprocess(pd.DataFrame({"text":text}, index = [0]))
      o = self.create_token(o["text"].tolist())
      o = self.token2id(o)
      o = [self.cls_token_id] + o[0] + [self.sep_token_id]

      return o, [1]*len(o)

    def preprocess(self, df):
        df['text'] = df['text'].str.lower()
        df['text'] = df['text'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # remove emails
        df['text'] = df['text'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # remove IP address
        df['text'] = df['text'].str.replace('[^\w\s]','')                                                       # remove special characters
        df['text'] = df['text'].replace('\d', '', regex=True)
        return df
        
    def create_dataset(self,text,label):
        assert len(text) == len(label)
        tokenized_text = self.create_token(text)
        tokenized_text = self.token2id(tokenized_text)
        padmask = []
        for i,observation in enumerate(tokenized_text):
            maskedlength = 0
            if len(observation) == (self.max_length-2):
                tokenized_text[i] = [self.cls_token_id] + tokenized_text[i] + [self.sep_token_id]
            else:
                pad = [self.pad_token_id] * (self.max_length - len(observation) - 2)
                maskedlength = len(pad)
                tokenized_text[i] = [self.cls_token_id] + tokenized_text[i]+[self.sep_token_id] + pad 
            padmask.append([1]*(len(observation)+2) + [0]*(maskedlength))
        
        database = TensorDataset(torch.tensor(tokenized_text, dtype=torch.long),
                                 torch.tensor(label, dtype=torch.float),
                                 torch.tensor(padmask, dtype=torch.long))                              
        return database

class GPT2TextProcessor():
  def __init__(self, tokenizer, max_length = 100):
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.speaker1_token = "<SPEAKER1>"
    self.speaker2_token = "<SPEAKER2>"
    self.speaker1_token_id = self.tokenizer.convert_tokens_to_ids(self.speaker1_token)
    self.speaker2_token_id = self.tokenizer.convert_tokens_to_ids(self.speaker2_token)

  def create_dataset(self,conversations):
    pad = lambda x,y: x + [y]*(self.max_length - len(x))
    words = []
    segments = []
    labels = []
    attention_masks = []
    mc_token_ids = []
    for conversation in conversations:
      tokenized_line = [self.tokenizer.bos_token_id]
      line_segment = [self.speaker1_token_id]
      total_length = 0
      for i,line in enumerate(conversation):
        speaker = self.speaker1_token if i % 2 == 0 else self.speaker2_token
        text_line = speaker + " " + line["text"].replace("\n","")
        tokenized_text = tokenizer.tokenize(text_line)
        tokenized_line = tokenized_line + tokenizer.convert_tokens_to_ids(tokenized_text)
        line_segment = line_segment + [tokenizer.convert_tokens_to_ids(speaker)]*len(tokenized_text)
        last_seq_length = len(tokenized_text)
        total_length += last_seq_length
      if total_length >= self.max_length-1:
        continue


      tokenized_line = tokenized_line + [self.tokenizer.eos_token_id]
      label_mask = len(tokenized_line)-last_seq_length
      line_segment = line_segment + [line_segment[-1]]
      label_line = [-1]*(label_mask)+tokenized_line[label_mask:]
      labels.append(pad(label_line,-1))
      words.append(pad(tokenized_line,self.tokenizer.pad_token_id))
      segments.append(pad(line_segment,self.tokenizer.pad_token_id))
      mc_token_ids.append(len(tokenized_line)-1)
    dataset = TensorDataset(torch.tensor(words, dtype=torch.long),
                              torch.tensor(segments, dtype=torch.long),
                              torch.tensor(labels, dtype=torch.long))
    return dataset

class TransformerClassification(nn.Module):
    def __init__(self,transformerModel,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        super().__init__()
        self.transformerModel = transformerModel
        self.embedding_dim = transformerModel.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(self.embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,text,mask):
      with torch.no_grad():
        embeddings = self.transformerModel(input_ids = text,attention_mask = mask)[0]
      output, hidden = self.rnn(embeddings)
      if self.rnn.bidirectional:
          hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
      else:
          hidden = self.dropout(hidden[-1,:,:])
              
      #hidden = [batch size, hid dim]
      output = self.out(hidden)
      #output = [batch size, out dim]
      
      return output

def testSentiment(text, text_processor):
  model_bert.eval()
  prepared_text, mask = text_processor_bert.processString(text)
  prediction = model_bert(torch.tensor(prepared_text).view(1,-1).to(device),torch.tensor(mask).view(1,-1).to(device))
  if torch.round(torch.sigmoid(prediction)).item() == 0:
    print("NEGATIVE, with weight: ", torch.sigmoid(prediction).item())
  else:
    print("POSITIVE, with weight: ", torch.sigmoid(prediction).item())

class Conversation():
  def __init__(self, sentiment_model_path, text_model_path):
    """
    Conversation instance that utilizes a sentiment and text model.

    Args:
        sentiment_model_path (str): Path to the BERT-based sentimental analysis model.
        text_model_path (str): Path to the GPT2-based text generation model.

    """
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print('Using device:', device)

    self.tokenizer_bert = BertTokenizer.from_pretrained('bert-large-uncased')
    self.tokenizer_bert.pad_token = "<pad>"
    self.tokenizer_bert.sep_token = "[SEP]"
    self.tokenizer_bert.cls_token = "[CLS]"
    self.text_processor_bert = BertTextProcessor(self.tokenizer_bert)
  
    transformerModel = BertModel.from_pretrained("bert-large-uncased").to(self.device)
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25
    self.model_bert = TransformerClassification(transformerModel,
                                HIDDEN_DIM,
                                OUTPUT_DIM,
                                N_LAYERS,
                                BIDIRECTIONAL,
                                DROPOUT).to(self.device)

    self.model_bert.load_state_dict(torch.load(sentiment_model_path, map_location=torch.device(self.device)))

    SPECIAL_TOKENS = {"bos_token":"<BOS>","eos_token":"<EOS>",
                                          "pad_token":"<PAD>",
                                          "additional_special_tokens":("<SPEAKER1>","<SPEAKER2>")}
    self.model_gpt2, self.tokenizer_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device), GPT2Tokenizer.from_pretrained("gpt2")
    n_added_tokens = self.tokenizer_gpt2.add_special_tokens(SPECIAL_TOKENS)
    self.model_gpt2.resize_token_embeddings(new_num_tokens=self.tokenizer_gpt2.vocab_size+n_added_tokens)
    self.model_gpt2.load_state_dict(torch.load(text_model_path, map_location=torch.device(self.device)))
    
    self.model_bert.eval()
    self.model_gpt2.eval()
    self.history = [self.tokenizer_gpt2.bos_token_id]
    self.temperature = .5
    self.length = 60
    self.k = 100

  def reset(self):
    """
    Resets the conversation history.
    """
    self.history = [self.tokenizer_gpt2.bos_token_id]

  def next_sentence(self, input_sentence):
    """
    Returns an answer and sentiment prediction based on an input sentence.

    Args:
        input_sentence (str): The first parameter.

    Returns:
      (tuple): tuple containing:
        decoded_answer (str): Answer to input_sentence.
        prediction_float (float): Sentiment prediction, 0 = Negative, 1 = Positive.
    """
    sequence = self.tokenizer_gpt2.encode(input_sentence,add_special_tokens=False) + self.tokenizer_gpt2.convert_tokens_to_ids(["<SPEAKER2>"])

    answer = []
    input_ids = self.history + sequence
    for i in range(self.length):
      with torch.no_grad():
        out = self.model_gpt2(torch.tensor(input_ids).view(1,-1).to(self.device),out[1] if i >0 else None)
      logits = out[0][:, -1, :] / (self.temperature if self.temperature >0 else 1)
      indices_to_remove = logits < torch.topk(logits, self.k)[0][..., -1, None]
      logits[indices_to_remove] = -float('Inf')
      
      if self.temperature == 0: # greedy sampling:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
      else:
        next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
      
      if next_token[0].item() == self.tokenizer_gpt2.eos_token_id: break
      
      answer.append(next_token[0].item())
      input_ids = next_token
    decoded_answer = self.tokenizer_gpt2.decode(answer)
    prepared_text, mask = self.text_processor_bert.processString(decoded_answer)
    prediction = self.model_bert(torch.tensor(prepared_text).view(1,-1).to(self.device),torch.tensor(mask).view(1,-1).to(self.device))
    prediction_float = torch.sigmoid(prediction).item()
    self.history = self.history + sequence + answer + self.tokenizer_gpt2.convert_tokens_to_ids(["<SPEAKER1>"])
    return decoded_answer, prediction_float

if __name__ == "__main__":
  conv = Conversation('models/bestModelBert.pt', 'models/bestModelGPT2CONV.pt')
  while True:
    raw_text = input(">>> ")
    if raw_text == "quit": 
      break
    elif raw_text == "reset":
      print("Removing history...")
      conv.reset()
    else:
      answer, sentiment = conv.next_sentence(raw_text)
      print(answer , {sentiment > .9: ':)', sentiment < .1: ':('}.get(True, ""))
