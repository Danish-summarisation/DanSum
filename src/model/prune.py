'''
This script contains the code for creating a compressed version of mT5, containing only embeddings for the most used Danish and English vocabulary.

Code adapted from https://gist.github.com/avidale/44cd35bfcdaf8bedf51d97c468cc8001.
'''

# run git lfs install from terminal

from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm, trange

# loading model and tokeniser
tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained('google/mt5-small')

# loading and preparing corpora
da_gigaword = load_dataset("DDSC/dagw_reddit_filtered_v1.0.0")
en_gigaword = load_dataset("gigaword")
danewsroom = pd.read_csv("/data-big-projects/danish-summarization-danewsroom/train_all_.csv", usecols=['text'])

df_en_train = pd.DataFrame(en_gigaword['train'])
df_en_val = pd.DataFrame(en_gigaword['validation'])
df_en_test = pd.DataFrame(en_gigaword['test'])
df_en = pd.concat([df_en_train, df_en_val, df_en_test])

df_da_gigaword = pd.DataFrame(da_gigaword['train'], columns=['text'])
df_da = pd.concat([df_da_gigaword, danewsroom])

cnt_da = Counter()
for text in tqdm(df_da.text):
   cnt_da.update(tokenizer.encode(text))

cnt_en = Counter()
for text in tqdm(df_en.document):
    cnt_en.update(tokenizer.encode(text))

print('Tokenised Danish words:', len(cnt_da))
print('Tokenised English words:', len(cnt_en))
common = len(set(cnt_da.keys()).intersection(set(cnt_en.keys())))
print('Common word between corpora:', common)
print('Amount of Danish words that are also in English corpora:', common / len(cnt_da)*100)

diff_en = len(set(cnt_en.keys()).difference(set(cnt_da.keys())))
print('Words that are only in the English corpus:', diff_en)
diff_da = len(set(cnt_da.keys()).difference(set(cnt_en.keys())))
print('Words that are only in the Danish corpus:', diff_da)

total = common + diff_en + diff_da
print('Total number of tokenised words across corpora:', total)

print('Percentage of total model vocabulary:', total/tokenizer.vocab_size*100)

print('Danish top tokens')
for top in 10_000, 20_000, 30_000:
    print(top, sum(v for k, v in cnt_da.most_common(top)) / sum(cnt_da.values()))
print('English top tokens')
for top in 10_000, 20_000, 30_000:
    print(top, sum(v for k, v in cnt_en.most_common(top)) / sum(cnt_en.values()))

# removing unused embeddings
old_voc = tokenizer.get_vocab()
old_inv_voc = {v: k for k, v in old_voc.items()}

print('Danish:', tokenizer.convert_ids_to_tokens([k for k, v in cnt_da.most_common(30)]))
print('English:', tokenizer.convert_ids_to_tokens([k for k, v in cnt_en.most_common(30)]))


new_tokens = set(range(1000))
for i, (k, v) in enumerate(cnt_en.most_common(10_000)):
    if k not in new_tokens:
        new_tokens.add(k)
for i, (k, v) in enumerate(cnt_da.most_common(25_000)):
    if len(new_tokens) == 29_900:
        print(i, 'Danish tokens are included')
        break
    if k not in new_tokens:
        new_tokens.add(k)

for t in range(tokenizer.vocab_size - 100, tokenizer.vocab_size):
    new_tokens.add(t)

print(len(new_tokens))
kept_ids = sorted(new_tokens)

len(kept_ids) / tokenizer.vocab_size

# updating embeddings
new_size = len(kept_ids)
new_emb = torch.nn.Embedding(new_size, model.shared.embedding_dim)
new_head = torch.nn.Linear(in_features=model.lm_head.in_features, out_features=new_size, bias=False)

for new_id, old_id in enumerate(kept_ids):
    new_emb.weight.data[new_id] = model.shared.weight.data[old_id]
    new_head.weight.data[new_id] = model.lm_head.weight.data[old_id]

model.shared.weight = new_emb.weight
model.lm_head.weight = new_head.weight

#!wget https://raw.githubusercontent.com/google/sentencepiece/master/src/sentencepiece_model.proto

#!sudo apt install protobuf-compiler

#!protoc --python_out=. sentencepiece_model.proto

import sentencepiece_model_pb2 as spmp
smp = tokenizer.sp_model.serialized_model_proto()
m = spmp.ModelProto()
m.ParseFromString(smp)

print('the loaded model has pieces:', len(m.pieces))
new_pieces = [m.pieces[idx] for idx in kept_ids]
print('the new pieces:', len(new_pieces))

# replace the content of the first 30K pieces
for i, p in enumerate(new_pieces):
    m.pieces[i].piece = p.piece
    m.pieces[i].score = p.score
    m.pieces[i].type = p.type

# drop the remaining pieces
n = len(new_pieces)
for i in trange(len(m.pieces) - n):
    m.pieces.pop(len(m.pieces) - 1)

print(len(m.pieces))
with open('new_sp.model', 'wb') as f:
    f.write(m.SerializeToString())

new_tokenizer = T5Tokenizer('new_sp.model', extra_ids=0)

# save model
model.config.__dict__['vocab_size'] = new_size
model.config.__dict__['_name_or_path'] = 'cointegrated/daT5-base'
model.config

new_tokenizer.save_pretrained('daT5-base')
model.save_pretrained('daT5-base')