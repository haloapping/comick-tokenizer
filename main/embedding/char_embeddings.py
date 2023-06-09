import pickle
import numpy as np

with open("C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/main/char_mimick_glove_d100_c20", encoding="UTF-8") as f:
    chars_embedding = f.readlines()

chars_embedding = [embedding.split("\n") for embedding in chars_embedding]
chars_embedding = [embedding[0].split(" ") for embedding in chars_embedding]

chars_embedding = np.array(chars_embedding)
chars = chars_embedding[:, 0]
embeddings = chars_embedding[:, 1:].astype(np.float32)
char_embeddings = {char: embedding for char, embedding in zip(chars, embeddings)}
# print(char_embeddings)

file = open("C:/Users/62821/Google Drive (if-17041@students.ithb.ac.id)/raja_terakhir/thesis/apps/oov-data-preprocesing/main/char_embeddings.pkl", "ab")
pickle.dump(char_embeddings, file)
