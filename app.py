import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")
st.write("Content-based recommender using NLP & cosine similarity")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    credits = pd.read_csv("credits.csv")

    movies = movies.merge(credits, on="title")
    movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
    return movies

movies = load_data()

# -------------------- PREPROCESSING --------------------
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def convert3(obj):
    L = []
    for i in ast.literal_eval(obj)[:3]:
        L.append(i['name'])
    return L

def fetch_director(obj):
    return [i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director']

movies.dropna(inplace=True)

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

for col in ['genres','keywords','cast','crew']:
    movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = (
    movies['overview'] +
    movies['genres'] +
    movies['keywords'] +
    movies['cast'] +
    movies['crew']
)

movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

new_df = movies[['movie_id','title','tags']]

# -------------------- STEMMING --------------------
ps = PorterStemmer()

def stem(text):
    return " ".join(ps.stem(word) for word in text.split())

new_df['tags'] = new_df['tags'].apply(stem)

# -------------------- VECTORIZATION --------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

# -------------------- RECOMMENDER --------------------
def recommend(movie):
    movie = movie.lower()

    matches = new_df[new_df['title'].str.lower().str.contains(movie)]

    if matches.empty:
        return None

    index = matches.index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    results = []
    for i in movie_list:
        results.append((
            new_df.iloc[i[0]].title,
            round(i[1], 3)
        ))

    return results


# -------------------- STREAMLIT UI --------------------
selected_movie = st.selectbox(
    "Select a movie",
    new_df['title'].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    st.subheader("🎬 Recommended Movies")

    for movie, score in recommendations:
      st.markdown(f"""
      **{movie}**  
      
      """)


  

