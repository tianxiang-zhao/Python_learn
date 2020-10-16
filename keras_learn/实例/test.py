from opt_einsum.backends import torch
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig

tokenizer =BertWordPieceTokenizer("bert-base-uncased/vocab.txt", lowercase=True)
model = BertTokenizer.from_pretrained('bert-base-uncased')
sentence = "Hello, my son is cuting."
tokenizer_context = tokenizer.encode(sentence, add_special_tokens=True)  # Batch size 1
    # tensor([ 101, 7592, 1010, 2026, 2365, 2003, 3013, 2075, 1012,  102])
print(tokenizer_context)
# Find tokens that were created from answer characters
for idx, (start, end) in enumerate(tokenizer_context.offsets):
    print(idx,start,end)

context = " ".join(str("Hello, my son is cuting").split())
print(context)
print(len(context))
print("tokenizer_context.ids",tokenizer_context.ids)
print([0]*5+[1]*3)