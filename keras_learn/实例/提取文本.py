import os
import re
import json
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig

max_len=384
configuretion=BertConfig()#BERT的默认参数和配置 default parameters and configuration for BERT

#Set-up BERT tokenizer #分词器
slow_tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
save_path="bert-base-uncased/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer("bert-base-uncased/vocab.txt", lowercase=True)
#Load the data
train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
train_path = keras.utils.get_file("train.json", train_data_url)
eval_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
eval_path = keras.utils.get_file("eval.json", eval_data_url)

class SquadExample:
    def __init__(self,quesition,context,start_char_idx,anwser_text,all_anwser):
        self.question=quesition
        self.context=context
        self.start_char_idx=start_char_idx
        self.anwser_text=anwser_text
        self.all_anwser=all_anwser
        self.skip=False

    def preprocess(self):
        context=self.context
        question=self.question
        anwser=self.anwser_text
        start_char_ids=self.start_char_idx

        # Clean context, answer and question
        context=" ".join(str(context).split())
        question=" ".join(str(question).split())
        anwser=" ".join(str(anwser).split())

        # Find end character index of answer in context
        end_char_idx=self.start_char_idx+len(anwser)
        if end_char_idx >=len(context):
            self.skip=True
            return

        # Mark the character indexes in context that are in answer  在应答的上下文中标记字符索引
        is_char_in_ans=[0]*len(context)
        for idx in range(start_char_ids,end_char_idx):
            is_char_in_ans[idx]=1

        # Tokenize context
        tokenizer_context=tokenizer.encode(context)

        # Find tokens that were created from answer characters 查找从应答字符创建分词
        ans_token_idx = []
        for idx,(start,end) in enumerate(tokenizer_context.offsets):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)

        if len(ans_token_idx) == 0:
            self.skip = True
            return

        # Find start and end token index for tokens from answer
        start_token_idx=ans_token_idx[0]
        end_token_idx=ans_token_idx[-1]

        # Tokenize question
        tokenized_question = tokenizer.encode(question)

        # Create inputs
        inputs_ids=tokenizer_context.ids+tokenized_question.ids[1:]
        token_type_ids=[0]*len(tokenizer_context.ids)+[1]*len(tokenized_question.ids[1:])
        attention_mask=[1]*inputs_ids

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_lengs=max_len-len(inputs_ids)
        if padding_lengs>0:
            inputs_ids=inputs_ids+([0]*padding_lengs)
            token_type_ids=token_type_ids+([0]*padding_lengs)
            attention_mask=attention_mask+([0]*padding_lengs)
        elif padding_lengs<0:
            self.skip=True
            return

        self.input_ids = inputs_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.context_token_to_char = tokenizer_context.offsets

with open(train_path) as f:
    raw_train_data=json.load(f)
with open(eval_path) as f:
    raw_aval_data=json.load(f)

def create_squad_example(raw_data):
    squad_example=[]
    for item in raw_data["data"]:
        for para in item["paragraphs"]:
            context=para["context"]
            for qa in para["qas"]:
                question=qa["question"]
                anwser_text=qa["anwser"][0]["text"]
                all_anwser=[_["text"] for _ in qa["anwser"]]
                start_char_idx=qa["anwser"][0]["anwser_start"]

                squad_eg=SquadExample(question,context,start_char_idx,anwser_text,all_anwser)

                squad_eg.preprocess()
                squad_example.append(squad_eg)
    return squad_example




