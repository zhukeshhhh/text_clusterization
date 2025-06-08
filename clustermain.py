#!/usr/bin/env python
# coding: utf-8

# %memit
from loguru import logger

import torch
from sentence_transformers import SentenceTransformer

from transformers import BertTokenizer, BertModel, BertConfig
import time


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np


import re
import string
import nltk
from nltk.corpus import stopwords

# viz libs
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt_tab')


categories = [
 'biology',
 'cooking',
 'diy',
 'robotics',
 'travel',
 'cryptography'
]

df1 = pd.read_csv('biology.csv')
df2 = pd.read_csv('cooking.csv')
df3 = pd.read_csv('diy.csv')
df4 = pd.read_csv('robotics.csv')
df5 = pd.read_csv('travel.csv')
df6 = pd.read_csv('crypto.csv')

df1['target'] = 0
df2['target'] = 1
df3['target'] = 2
df4['target'] = 3
df5['target'] = 4
df6['target'] = 5


df1 = df1.iloc[0:300]
df2 = df2.iloc[0:300]
df3 = df3.iloc[0:300]
df4 = df4.iloc[0:300]
df5 = df5.iloc[0:300]
df6 = df6.iloc[0:300]
logger.success("Data loaded successfully")
#df = pd.concat([df1, df2, df3, df4, df5, df6])
df = pd.concat([df1, df2, df3, df4, df5, df6])
# device = torch_directml.device()
# device = torch.device("ipu")
# Uncomment for apple ARM processors
# device = torch.device("mps")

# Uncomment for amd on windoes
try:
    import torch_directml
    device = torch_directml.device()
except Exception as e:
    logger.exception(e)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logger.info(f"Using device: {device}")


def preprocess_text(text: str) -> str:
    # remove html tags
    text = re.sub(r"<[^>]*>", " ", text)
    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)

    # remove stopwords
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
    text = " ".join(tokens)
    text = text.lower().strip()

    return text


from tqdm import tqdm
tqdm.pandas()
logger.info("Preprocessing text...")
# Если кто-то придумает как это кешировать или распаралелить - куплю шоколадку.
df['text_cleaned'] = df['content'].progress_apply(preprocess_text)
df = df[df['text_cleaned'] != '']


#df['len_words'].describe()


# ### TF-IDF Vectorization
# 
# This is a simple but effective method for generating vector representations of sentences. It stands for "term frequency-inverse document frequency" and it calculates the importance of words in a sentence by taking into account how often they appear in the sentence and how rare they are in the entire corpus of sentences. 
# 
# 
logger.info("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
X = vectorizer.fit_transform(df['text_cleaned']).toarray()


# Cluster 0 refers to sport, cluster 2 to software / tech, cluster 3 to religion

# ### Sentence Transformer
# Sentence Transformers are deep learning models that can encode natural language sentences into high-dimensional vector representations. They are trained using a pre-training and fine-tuning approach and have achieved state-of-the-art performance on several natural language processing tasks. These models are widely used for various applications such as chatbots, search engines, and recommendation systems.

tqdm.pandas()

st = time.time()
modelNameSentence = 'all-MiniLM-L12-v2'
logger.info("Vectorizing text with Sentence Transformers using model: {}".format(modelNameSentence))
model = SentenceTransformer(modelNameSentence, device=device)
df['encode_transforemers'] = df['text_cleaned'].progress_apply(lambda text: model.encode(text, convert_to_numpy=True).flatten())

et = time.time()

logger.info("Elapsed time: {:.2f} seconds".format(et - st))


model.to(device = device)


# 

X_transformers = np.vstack(df['encode_transforemers'])


# ### BERT - [CLS] token for sentnce context
# 
# BERT, is a pre-trained deep learning model that can be fine-tuned for various natural language processing tasks. One of the main innovations of BERT is its ability to represent both the left and right context of a word, allowing it to better capture the meaning of a sentence.
# 
# In BERT, the [CLS] token, which stands for "classification", is a special token that is inserted at the beginning of every input sequence. During pre-training, BERT is trained to predict the correct class label for the entire sequence based on the [CLS] token representation, which is meant to capture the overall meaning of the sequence.

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from transformers import pipeline
pipeline = pipeline(
    task="fill-mask",
    model="google-bert/bert-base-uncased",
    torch_dtype=torch.float,
    device=device,
)


def check_tensor_count():
    import gc
    count = 0
    for obj in gc.get_objects():

        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                count += 1

        except:
            pass
    print("Tensor count:", count)

input_ids = None
def get_cls_sentence(sentence):
    global input_ids
    # Tokenize input sentence and convert to tensor
    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True, max_length=512)])

    if device == torch_directml.device():
        input_ids = input_ids.to(device=device)





    # Pass input through BERT model and extract embeddings for [CLS] token
    with torch.no_grad():
        return pipeline.model(input_ids)[0][:, 0, :].flatten()





# 
check_tensor_count()
st = time.time()
tqdm.pandas()
df['cls_bert'] = df['text_cleaned'].progress_apply(lambda sentence: get_cls_sentence(sentence))

et = time.time()

print("Elapsed time: {:.2f} seconds".format(et - st))

check_tensor_count()
X_cls_bert = np.vstack(df['cls_bert'])


df[df['target'] == 2]['text_cleaned'].value_counts()


# ### Clustering
# #### 1. K-Means

# To evaluate the performance of a clustering algorithm like k-means, we use various metrics that compare the predicted clusters to the ground truth labels. Here are a few common metrics:
# 
# Adjusted Rand Index (ARI): measures the similarity between the predicted clusters and the ground truth labels, taking into account chance agreement. ARI ranges from -1 to 1, where 1 indicates perfect agreement and 0 indicates random clustering.
# 
# Normalized Mutual Information (NMI): measures the mutual information between the predicted clusters and the ground truth labels, normalized by the entropy of the clusters and labels. NMI ranges from 0 to 1, where 1 indicates perfect agreement.
# 
# Fowlkes-Mallows Index (FMI): measures the geometric mean of the precision and recall of the predicted clusters with respect to the ground truth labels. FMI ranges from 0 to 1, where 1 indicates perfect agreement.

def eval_cluster(embedding, kmeans):
    y_pred = kmeans.fit_predict(embedding)

    # Evaluate the performance using ARI, NMI, and FMI
    ari = adjusted_rand_score(df["target"], y_pred)
    nmi = normalized_mutual_info_score(df["target"], y_pred)
    fmi = fowlkes_mallows_score(df["target"], y_pred)

    # Print Metrics scores
    print("Adjusted Rand Index (ARI): {:.3f}".format(ari))
    print("Normalized Mutual Information (NMI): {:.3f}".format(nmi))
    print("Fowlkes-Mallows Index (FMI): {:.3f}".format(fmi))


def dimension_reduction(embedding, method):

    pca = PCA(n_components=2, random_state=42)

    pca_vecs = pca.fit_transform(embedding)

    # save our two dimensions into x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]

    df[f'x0_{method}'] = x0 
    df[f'x1_{method}'] = x1 



def plot_pca(x0_name, x1_name, cluster_name, method):

    plt.figure(figsize=(12, 7))

    plt.title(f"TF-IDF + KMeans 20newsgroup clustering with {method}", fontdict={"fontsize": 18})
    plt.xlabel("X0", fontdict={"fontsize": 16})
    plt.ylabel("X1", fontdict={"fontsize": 16})

    sns.scatterplot(data=df, x=x0_name, y=x1_name, hue=cluster_name, palette="viridis")
    plt.show()


for embedding_and_method in [(X, 'tfidf'), (X_transformers, 'transformers')]:
    embedding, method = embedding_and_method[0], embedding_and_method[1]

    # initialize kmeans with 3 centroids
    #kmeans = KMeans(n_clusters=6, random_state=42)
    kmeans = KMeans(n_clusters=6, random_state=42)
    # fit the model
    kmeans.fit(embedding)

    # store cluster labels in a variable
    clusters = kmeans.labels_

    # Assign clusters to our dataframe
    clusters_result_name = f'cluster_{method}'
    df[clusters_result_name] = clusters

    eval_cluster(embedding, kmeans)

    dimension_reduction(embedding, method)

    plot_pca(f'x0_{method}', f'x1_{method}', cluster_name=clusters_result_name, method=method)


df


# ### PCA & Vizualization
