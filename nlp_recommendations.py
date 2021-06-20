import pandas as pd
import re
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stop_words_ = set(nltk.corpus.stopwords.words('english'))


def isnan(value):
    try:
        import math
        return math.isnan(float(value))
    except:
        return False


def black_txt(token):
    return token not in stop_words_ and token not in list(string.punctuation) and len(token) > 2


def clean_text(text):
    clean_text = []
    clean_text2 = []
    text = re.sub("'", "", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    clean_text = [nltk.stem.WordNetLemmatizer().lemmatize(word, pos="v")
                  for word in nltk.word_tokenize(text.lower()) if black_txt(word)]
    clean_text2 = [word for word in clean_text if black_txt(word)]
    return " ".join(clean_text2)


def generate_final_result(df_result, df_songs_original):
    final_result = []
    for index in range(len(df_result)):
        song_df = df_songs_original.loc[df_songs_original['_id'] ==
                                        df_result.iloc[index]['songId']]
        song_df = song_df.reset_index(drop=True)
        song_dict = {
            "_id": song_df['_id'].values[0],
            "name": song_df['name'].values[0],
            "genre": {
                '_id': song_df['genre._id'].values[0],
                'name': song_df['genre.name'].values[0]
            },
            "artist": {
                '_id': song_df['artist._id'].values[0],
                'name': song_df['artist.name'].values[0]
            },
            "image": song_df['image'].values[0] or ''
        }

        if isnan(song_dict["image"]):
            song_dict["image"] = 'https://www.shutterstock.com/image-vector/music-notes-song-melody-tune-flat-701307613'

        songs_json = json.dumps(song_dict, indent=4)
        final_result.append(songs_json)
    return final_result


def get_user_recommendations(user_id):
    songsCsv = pd.read_csv('songs.csv')
    df_songs_original = pd.DataFrame(songsCsv, columns=[
        "_id", "name", "genre._id", "genre.name", "artist._id", "artist.name", "image"
    ])

    songsCsv.rename(columns={
        '_id': 'songId',
        'genre.name': 'genreName',
        'artist.name': 'artistName',
    }, inplace=True)

    df_songs = pd.DataFrame(songsCsv, columns=[
        'songId', 'name', 'artistName', 'genreName', 'userId'])
    df_songs = df_songs.fillna(" ")

    ratingsCsv = pd.read_csv('rating.csv')
    ratingsCsv.rename(columns={
        '_id': 'songId',
        'genre.name': 'genreName',
        'artist.name': 'arti stName',
    }, inplace=True)

    df_history = pd.DataFrame(ratingsCsv, columns=[
        'songId', 'name', 'artistName', 'genreName', 'userId', 'text'])
    df_history = df_history.fillna(" ")

    df_history['text'] = df_history.groupby(
        ['userId'])['name', 'artistName', 'genreName'].transform(lambda x: ' '.join(x))
    df_history = df_history[['userId', 'text']].drop_duplicates(
        subset='userId', keep='first').reset_index()

    df_songs["text"] = df_songs["name"].map(
        str) + " " + df_songs["artistName"].map(str) + " " + df_songs["genreName"].map(str)
    df_songs["text"] = df_songs["text"].map(str).apply(clean_text)
    df_songs["text"] = df_songs["text"].str.lower()
    df_songs = df_songs[["songId", "text"]]

    df_history["text"] = df_history["text"].map(str).apply(clean_text)
    df_history["text"] = df_history["text"].str.lower()
    df_history = df_history[["userId", "text"]]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_songs = tfidf_vectorizer.fit_transform(df_songs['text'])

    count_vectorizer = CountVectorizer()
    count_songs = count_vectorizer.fit_transform(df_songs['text'])

    user = df_history.loc[df_history["userId"] == user_id]

    if user.empty:
        final_result = generate_final_result(
            songsCsv.head(50), df_songs_original)
        return final_result

    user_tfidf = tfidf_vectorizer.transform(user['text'])
    user_count = count_vectorizer.transform(user['text'])

    cos_similarity_tfidf = map(
        lambda x: cosine_similarity(user_tfidf, x), tfidf_songs)
    cos_similarity_countv = map(
        lambda x: cosine_similarity(user_count, x), count_songs)

    df_final_tfidf = pd.DataFrame(df_songs)
    df_final_tfidf['score'] = 0
    cos_similarity_tfidf = list(cos_similarity_tfidf)
    for i in range(len(cos_similarity_tfidf)):
        df_final_tfidf.loc[i, 'score'] = cos_similarity_tfidf[i][0]
    df_final_tfidf = df_final_tfidf.sort_values('score', ascending=False)
    df_final_tfidf.to_csv('tfidf.csv')

    df_final_count = pd.DataFrame(df_songs)
    df_final_count['score'] = 0
    cos_similarity_countv = list(cos_similarity_countv)
    for i in range(len(cos_similarity_tfidf)):
        df_final_count.loc[i, 'score'] = cos_similarity_countv[i][0]
    df_final_count = df_final_count.sort_values('score', ascending=False)
    df_final_count.to_csv('countv.csv')

    df_result = df_final_count

    if df_final_tfidf.loc[0, 'score'] > df_final_count.loc[0, 'score']:
        df_result = df_final_tfidf
    df_result = df_result.head(50)

    final_result = generate_final_result(df_result, df_songs_original)
    return final_result
