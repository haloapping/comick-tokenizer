import sys
import pandas as pd
import numpy as np
import pickle
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from ngrams.oov_ngrams import OOVNgrams
from ngrams.util import Util
from tqdm import tqdm
from itertools import chain


# Read dataset and add OOV flag
raw_df = pd.read_csv("C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/main/id_with_oov_flag.csv", skip_blank_lines=False)
tokens_with_oov_flag = list(raw_df[["token", "is_oov"]].itertuples(index=False, name=None))

# Tokens to docs
def make_sentence(tokens, return_sentences=False):
    sentence = []
    sentences = []

    for token in tqdm(tokens):
        if token[0] is not np.nan:
            sentence.append(token)
        else:
            sentences.append(sentence)
            sentence = []

    if return_sentences:
        return len(sentences), sentences 
    
    return len(sentences)

n_docs, docs = make_sentence(tokens_with_oov_flag, True)
print(len(docs))
print("Make sentences done 😉\n")

# OOV Ngrams Dataset
oov_ngrams = OOVNgrams()
context_size = 1
docs_with_oov = oov_ngrams.create_ngrams(docs, context_size=context_size)
file_name = open(f"C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/pickle/features/{79 if (context_size > 79) else context_size}_context/oov_ngrams.pkl", "ab")
pickle.dump(docs_with_oov, file_name)
print(len(docs_with_oov))
for doc in docs_with_oov[:5]:
    print(doc)
print("OOV Ngrams done 😉\n")

# Split OOV Ngrams
docs_split = oov_ngrams.split_ngrams(docs_with_oov, lowercase=True, split_token=True)
file_name = open(f"C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/pickle/features/{79 if (context_size > 79) else context_size}_context/split_oov_ngrams.pkl", "ab")
pickle.dump(docs_split, file_name)
for doc in docs_split[:5]:
    print(doc)
print("Split OOV Ngrams done 😉")

# left context
print("\nLeft context")
left_contexts = oov_ngrams.left_context(docs_with_oov)
for doc in left_contexts[:10]:
    print(doc)

# oov context split token
print("\nOOV context")
oov_contexts = oov_ngrams.oov_context(docs_with_oov, lowercase=True, split_token=True)
for doc in oov_contexts[:10]:
    print(doc)
print(len(oov_contexts))

# oov context not split token
print("\nOOV context")
oov_contexts_not_split = oov_ngrams.oov_context(docs_with_oov, lowercase=True, split_token=False)
for doc in oov_contexts_not_split[:10]:
    print(doc)
print(len(oov_contexts_not_split))
file_name = open(f"C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/pickle/features/{79 if (context_size > 79) else context_size}_context/lables.pkl", "ab")
pickle.dump(oov_contexts_not_split, file_name)
util = Util()
tokens = list(chain(*oov_contexts_not_split))
label_vocabs = util.vocabs(tokens)
label_vocabs = util.token_to_idx(label_vocabs)
file_name = open(f"C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/pickle/features/{79 if (context_size > 79) else context_size}_context/lable_vocabs.pkl", "ab")
pickle.dump(label_vocabs, file_name)

# right context
print("\nRight context")
right_contexts = oov_ngrams.right_context(docs_with_oov)
for doc in right_contexts[:10]:
    print(doc)

# padding left context
print("\nLeft context with padding")
util = Util()
pad_left_context = util.padding(left_contexts, mode="post", padding_val="<PAD>")
file_name = open(f"C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/pickle/features/{79 if (context_size > 79) else context_size}_context/left_context_with_pad.pkl", "ab")
pickle.dump(pad_left_context, file_name)
for doc in pad_left_context[:10]:
    print(doc)

# padding OOV context
print("\nOOV context with padding")
util = Util()
pad_oov_context = util.padding(oov_contexts, mode="post", padding_val='PAD')
file_name = open(f"C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/pickle/features/{79 if (context_size > 79) else context_size}_context/oov_context_with_pad.pkl", "ab")
pickle.dump(pad_oov_context, file_name)
for doc in pad_oov_context[:10]:
    print(doc)
print(len(pad_oov_context))

# padding right context
print("\nRight context with padding")
util = Util()
pad_right_context = util.padding(right_contexts, mode="post", padding_val='<PAD>')
file_name = open(f"C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/pickle/features/{79 if (context_size > 79) else context_size}_context/right_context_with_pad.pkl", "ab")
pickle.dump(pad_right_context, file_name)
for doc in pad_right_context[:10]:
    print(doc)

# Left context vocabs
print("\nLeft context Vocabs")
util = Util()
tokens = list(chain(*pad_left_context))
left_context_vocabs = util.vocabs(tokens)
file_name = open(f"C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/pickle/features/{79 if (context_size > 79) else context_size}_context/left_context_vocabs.pkl", "ab")
pickle.dump(left_context_vocabs, file_name)
print(left_context_vocabs[:50])
print(f"Number of left context vocab: {len(left_context_vocabs)}")

# Right context vocabs
print("\nRight context Vocabs")
util = Util()
tokens = list(chain(*pad_right_context))
right_context_vocabs = util.vocabs(tokens)
file_name = open(f"C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/pickle/features/{79 if (context_size > 79) else context_size}_context/right_context_vocabs.pkl", "ab")
pickle.dump(right_context_vocabs, file_name)
print(right_context_vocabs[:50])
print(f"Number of right context vocab: {len(right_context_vocabs)}")

# All context vocab
left_right_context_vocabs = sorted(list(set(left_context_vocabs + right_context_vocabs)))
print(f"All context vocabs: {left_right_context_vocabs[:10]}")
file_name = open(f"C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/pickle/features/{79 if (context_size > 79) else context_size}_context/left_right_vocabs.pkl", "ab")
pickle.dump(left_right_context_vocabs, file_name)
print(len(left_right_context_vocabs))

# OOV vocabs
print("\nOOV Vocabs")
util = Util()
tokens = list(chain(*oov_contexts))
oov_vocabs = util.vocabs(tokens)
file_name = open(f"C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/pickle/features/{79 if (context_size > 79) else context_size}_context/oov_context_vocabs.pkl", "ab")
pickle.dump(oov_vocabs, file_name)
print(oov_vocabs[:50])
print(f"Number of OOV vocab: {len(oov_vocabs)}")

# Vocab idx2token
left_context_idx2token = util.idx_to_token(left_context_vocabs)
file_name = open(f"C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/pickle/features/{79 if (context_size > 79) else context_size}_context/idx2token_left_context.pkl", "ab")
pickle.dump(left_context_idx2token, file_name)

oov_context_idx2token = util.idx_to_token(oov_vocabs)
file_name = open(f"C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/pickle/features/{79 if (context_size > 79) else context_size}_context/idx2token_oov_context.pkl", "ab")
pickle.dump(oov_context_idx2token, file_name)

right_context_idx2token = util.idx_to_token(right_context_vocabs)
file_name = open(f"C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/pickle/features/{79 if (context_size > 79) else context_size}_context/idx2token_right_context.pkl", "ab")
pickle.dump(right_context_idx2token, file_name)

print("\nLeft vocab dict:")
# print(left_context_idx2token)

# print("\nOOV vocab dict:")
# print(oov_context_idx2token)

# print("\nRight vocab dict:")
# print(right_context_idx2token)

# Vocab token2idx
left_context_token2idx = util.token_to_idx(left_context_vocabs)
file_name = open(f"C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/pickle/features/{79 if (context_size > 79) else context_size}_context/token2idx_left_context.pkl", "ab")
pickle.dump(left_context_token2idx, file_name)
# print(left_context_token2idx)

oov_context_token2idx = util.token_to_idx(oov_vocabs + ["PAD"])
file_name = open(f"C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/pickle/features/{79 if (context_size > 79) else context_size}_context/token2idx_oov_context.pkl", "ab")
pickle.dump(oov_context_token2idx, file_name)

right_context_token2idx = util.token_to_idx(right_context_vocabs)
file_name = open(f"C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/pickle/features/{79 if (context_size > 79) else context_size}_context/token2idx_right_context.pkl", "ab")
pickle.dump(right_context_token2idx, file_name)
print(len(left_context_idx2token), len(left_context_token2idx))

print("\nLeft vocab dict:")
# print(left_context_token2idx)

# print("\nOOV vocab dict:")
# print(right_context_token2idx)

# print("\nRight vocab dict:")
# print(oov_context_token2idx)

# max_left_context_size = max(len(doc) for doc in left_contexts)
# max_right_context_size = max(len(doc) for doc in right_contexts)
# print(max_left_context_size, max_right_context_size)
