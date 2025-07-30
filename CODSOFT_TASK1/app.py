import joblib
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from scipy.sparse import hstack


model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")  


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocessing(text):
    text = re.sub(r'\d+', '', text)
    text=text.lower()
    tokens=word_tokenize(text)
    filtered_tokens=[word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatized_tokens=[lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemmatized_tokens)


st.title("MOVIE GENRE CLASSIFICATION ")
st.write("Enter a movie description to predict its genre.")

description_input = st.text_area("Movie Description")

if st.button("Predict Genre"):
    if description_input:
        desc_clean = preprocessing(description_input)
        desc_vec = tfidf.transform([desc_clean])
        prediction = model.predict(desc_vec)
        st.success(f"Predicted Genre: {prediction[0]}")
    else:
        st.warning("Please provide description.")

