import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Carregando os dados dos animes
anime_df = pd.read_csv('Top_Anime_data (1).csv')

# Carregando a lista de stopwords
with open('stopwords.txt', 'r') as file:
    stopwords = set(word.strip() for word in file.readlines())

# Preprocessando os textos (removendo stopwords, pontuações e convertendo para minúsculas)
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())  # Remove caracteres especiais
    text = ' '.join(word for word in text.split() if word not in stopwords)  # Remove stopwords
    return text

anime_df['processed_description'] = anime_df['Description'].apply(preprocess_text)

# Criando a matriz TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(anime_df['processed_description'])

# Calculando a similaridade dos cossenos
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

# Normalizando as notas (Score) para a mesma escala que as similaridades
scaler = MinMaxScaler()
anime_df['normalized_score'] = scaler.fit_transform(anime_df[['Score']].fillna(0))

# Função de recomendação com peso para notas e similaridade
def recommend_animes_with_score(title, anime_df, similarity_matrix, score_weight=0.5, top_n=5):
    try:
        # Procurando o título parcial, ignorando maiúsculas e caracteres especiais
        normalized_title = re.sub(r'[^a-zA-Z\s]', '', title.lower())
        possible_matches = anime_df[anime_df['English'].fillna('').apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower())).str.contains(normalized_title)]
        if possible_matches.empty:
            return f"Anime '{title}' não encontrado na lista."
        
        # Selecionando o primeiro resultado como o título base
        idx = possible_matches.index[0]
        title = anime_df.loc[idx, 'English']
        
    except IndexError:
        return f"Anime '{title}' não encontrado na lista."
    
    # Incluindo o próprio anime na lista de recomendações
    recommendations = []
    sim_scores = list(enumerate(similarity_matrix[idx]))
    for i, sim in sim_scores:
        if i != idx:  # Ignorar o próprio anime
            recommendations.append((anime_df.iloc[i]['English'], float(sim), anime_df.iloc[i]['Score']))
    
    # Ordenando pelos mais similares
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Imprimindo resultados formatados como tabela
    print(f"\nRecomendações para: {title}")
    print("{:<50} {:<12} {:<6}".format("Anime", "Similarity", "Score"))
    print("-" * 70)
    for anime, similarity, score in recommendations:
        print(f"{anime:<50} {similarity:<12.4f} {score:<6.2f}")
    
    return recommendations

# Exemplo de uso "Altere o nome Mushoku Tensei ou mantenha-o para realizar a pesquisa"
recommend_animes_with_score('Mushoku Tensei', anime_df, cosine_sim_matrix, score_weight=0.7, top_n=20)
