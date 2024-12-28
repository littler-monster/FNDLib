import os
import pandas as pd
import numpy as np
from conf.config import Config

cfg = Config()


# Step 1: Load GloVe word embeddings
def load_glove_embeddings(glove_path, txt_path):
    embeddings_index = {}
    with open(os.path.join(glove_path, txt_path), encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Total %s word vectors.' % len(embeddings_index))
    return embeddings_index


# Step 2: Process text into word vectors
def preprocess_sentence(sentence, embeddings_index, max_length=30, embedding_dim=100):
    words = sentence.split()  # Tokenize
    embeddings = []

    for word in words[:max_length]:  # Truncate
        embedding = embeddings_index.get(word, [0.0] * embedding_dim)  # Get word vector; use zeros if not found
        embeddings.append(embedding)

    while len(embeddings) < max_length:  # Padding
        embeddings.append([0.0] * embedding_dim)

    return np.array(embeddings)


def preprocess_news(news, embeddings_index, max_sents, max_length, embedding_dim):
    processed_news = []
    for article in news:
        sentences = article.split('.')  # Split by period
        embeddings = []
        for sentence in sentences[:max_sents]:  # Truncate
            embeddings.append(preprocess_sentence(sentence.strip(), embeddings_index, max_length, embedding_dim))
        while len(embeddings) < max_sents:  # Padding
            embeddings.append(np.zeros((max_length, embedding_dim)))
        processed_news.append(np.array(embeddings))
    return np.array(processed_news)


def preprocess_comments(comments, embeddings_index, max_comments, max_length, embedding_dim):
    processed_comments = []
    for comment in comments:
        embeddings = []
        for sentence in comment[:max_comments]:  # Truncate
            embeddings.append(preprocess_sentence(sentence, embeddings_index, max_length, embedding_dim))
        while len(embeddings) < max_comments:  # Padding
            embeddings.append(np.zeros((max_length, embedding_dim)))
        processed_comments.append(np.array(embeddings))
    return np.array(processed_comments)


# Step 3: Load TSV files and extract sentences, labels, and comments
def load_news_from_csv(tsv_file_1, tsv_file_2):
    comments_data = pd.read_csv(tsv_file_2, sep='\t', encoding='utf-8')
    comments_dict = comments_data.groupby('id')['content'].apply(list).to_dict()

    data = pd.read_csv(tsv_file_1, sep='\t', encoding='utf-8')
    news = np.array(data['content'].tolist())
    labels = np.array(data['2_way_label'].tolist())
    comments = np.array([comments_dict.get(row['id'], []) for index, row in data.iterrows()])

    return news, comments, labels


# Main function
def main():
    news_path = "Politifact//politic_news_pics.tsv"
    comments_path = "Politifact//politic_comment.tsv"

    # Define the parameters
    embedding_dim = 100
    max_sents = 1
    max_length = 30
    max_comments = 15

    # Load data
    news, comments, labels = load_news_from_csv(news_path, comments_path)

    # Load GloVe word embeddings
    embeddings_index = load_glove_embeddings(cfg.glove_path, cfg.glove_txt_path)

    # Process all news articles
    processed_news = preprocess_news(news, embeddings_index, max_sents, max_length, embedding_dim)

    # Process comments
    processed_comments = preprocess_comments(comments, embeddings_index, max_comments, max_length, embedding_dim)

    # Shuffle sentences and labels
    indices = np.arange(len(processed_news))
    np.random.shuffle(indices)
    processed_news = processed_news[indices]
    labels = np.array(labels)[indices]
    processed_comments = processed_comments[indices]

    # Save as .npy files
    np.save(news_path+'news_politifact_2.npy', processed_news)
    np.save(news_path+'comments_politifact_2.npy', processed_comments)
    np.save(news_path+'labels_politifact_2.npy', labels)

    print("Processed news, comments, and labels saved to 'news_politifact_0.npy', 'comments_politifact_0.npy', and 'labels_politifact_0.npy'.")

if __name__ == "__main__":
    main()
