import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

anime_df = pd.read_csv('Top_Anime_data (1).csv')

with open('stopwords.txt', 'r') as file:
    stopwords = set(word.strip() for word in file.readlines())

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    text = ' '.join(word for word in text.split() if word not in stopwords)
    return text

anime_df['processed_description'] = anime_df['Description'].apply(preprocess_text)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(anime_df['processed_description'])


cosine_sim_matrix = cosine_similarity(tfidf_matrix)


scaler = MinMaxScaler()
anime_df['normalized_score'] = scaler.fit_transform(anime_df[['Score']].fillna(0))


def recommend_animes_with_score(title, anime_df, similarity_matrix, score_weight=0.5, top_n=5):
    try:
        
        normalized_title = re.sub(r'[^a-zA-Z\s]', '', title.lower())
        possible_matches = anime_df[anime_df['English'].fillna('').apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower())).str.contains(normalized_title)]
        if possible_matches.empty:
            return f"Anime '{title}' não encontrado na lista."
        
        
        idx = possible_matches.index[0]
        title = anime_df.loc[idx, 'English']
        
    except IndexError:
        return f"Anime '{title}' não encontrado na lista."
    
    
    recommendations = []
    sim_scores = list(enumerate(similarity_matrix[idx]))
    for i, sim in sim_scores:
        if i != idx:
            recommendations.append((anime_df.iloc[i]['English'], float(sim), anime_df.iloc[i]['Score']))
    
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
    
    print(f"\nRecomendações para: {title}") 
    print("{:<50} {:<12} {:<6}".format("Anime", "Similarity", "Score"))
    print("-" * 70)
    for anime, similarity, score in recommendations:
        print(f"{anime:<50} {similarity:<12.4f} {score:<6.2f}")
    
    return recommendations

anime_name = input("Digite o nome do anime para recomendação: ")
recommend_animes_with_score(anime_name, anime_df, cosine_sim_matrix, score_weight=0.7, top_n=20)
