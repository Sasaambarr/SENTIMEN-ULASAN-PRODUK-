
import streamlit as st
import joblib

# Load model dan vectorizer
model = joblib.load('naive_bayes_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.title("Aplikasi Analisis Sentimen Review Produk")
st.write("Masukkan ulasan produk untuk memprediksi sentimen (positif, negatif, netral).")

user_input = st.text_area("Tulis ulasan produk di sini:")

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Mohon masukkan teks ulasan terlebih dahulu.")
    else:
        text_tfidf = tfidf.transform([user_input])
        pred = model.predict(text_tfidf)[0]
        sentiment = label_encoder.inverse_transform([pred])[0]
        st.success(f"Sentimen ulasan ini adalah: **{sentiment.upper()}**")
