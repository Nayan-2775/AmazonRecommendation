import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image

# Load the dataset
data = pd.read_csv('amazon_product.csv')

# Remove unnecessary columns
if 'id' in data.columns:
    data = data.drop('id', axis=1)

# Download NLTK tokenizer if not present
nltk.download('punkt')

# Define tokenizer and stemmer
stemmer = SnowballStemmer('english')
def tokenize_and_stem(text):
    # Use preserve_line=True to avoid sentence tokenization
    tokens = nltk.word_tokenize(text.lower(), preserve_line=True)
    stems = [stemmer.stem(token) for token in tokens if token.isalpha()]
    return stems


# Create stemmed tokens column
data['stemmed_tokens'] = data.apply(lambda row: tokenize_and_stem(row['Title'] + ' ' + row['Description']), axis=1)

# Define TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)

# Streamlit UI
st.title("Amazon Product Recommender")

# Dropdown to select a product title
selected_title = st.selectbox("Select a product", data['Title'].values)

# Get stemmed tokens for the selected product
selected_tokens = data[data['Title'] == selected_title]['stemmed_tokens'].values[0]

# Calculate similarity with all products
data['similarity'] = data['stemmed_tokens'].apply(lambda x: cosine_similarity(
    tfidf_vectorizer.fit_transform([' '.join(selected_tokens), ' '.join(x)]))[0, 1])

# Sort and show top 5 similar products (excluding the selected one)
similar_products = data[data['Title'] != selected_title].sort_values(by='similarity', ascending=False).head(5)

st.subheader("Top 5 Similar Products:")
for i, row in similar_products.iterrows():
    st.markdown("**{}**".format(row['Title']))
    st.write(row['Description'])