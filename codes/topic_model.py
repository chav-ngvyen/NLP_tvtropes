import pandas as pd

from hdbscan import HDBSCAN
from bertopic import BERTopic

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from preprocess_functions import *

tdf = pd.read_csv("../TVTropesData/tropes.csv")

# Clean Trope column
tdf['Trope'] = clean_trope(tdf.Trope)
tdf = tdf.drop(columns="Unnamed: 0")

tdf['clean_description'] = preprocess(tdf.Description)



docs = tdf.clean_description

# Custom embedding on clean_description
sentence_model = SentenceTransformer("all-mpnet-base-v2")
embeddings = sentence_model.encode(docs, show_progress_bar=True)

# print(embeddings)

# print(type(embeddings))
# exit()

# HDBSCAN model
hdbscan_model = HDBSCAN(min_cluster_size=100, metric='euclidean', 
                        cluster_selection_method='eom',
                        prediction_data=True, min_samples=10)

# Vectorizer model
vectorizer_model = CountVectorizer(ngram_range=(1,1), stop_words="english")

# Train BERTopic
topic_model = BERTopic(
    verbose = True, 
    vectorizer_model = vectorizer_model,
    nr_topics = 'auto',
    hdbscan_model = hdbscan_model
    )

# Fit
topic_model.fit(docs, embeddings) 

# Save
topic_model.save("../models/bertopic")

exit()
