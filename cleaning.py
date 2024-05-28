import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pyarabic import araby

# Assurez-vous d'avoir téléchargé les stopwords arabes de NLTK
nltk.download('stopwords')

# Chargement du modèle et du vectorizer préalablement sauvegardés


# Définition des stopwords arabes de NLTK
nltk_arabic_stopwords = set(stopwords.words('arabic'))

# Fonction pour supprimer les stopwords arabes avec NLTK
def remove_nltk_stopwords(new_tweet):
    words = word_tokenize(new_tweet)  # Tokenisation du new_tweet
    filtered_words = [word for word in words if word not in nltk_arabic_stopwords]
    cleaned_tweet = ' '.join(filtered_words)  # Reconstitution du new_tweet nettoyé
    return cleaned_tweet

# Fonction pour nettoyer le new_tweet
def clean_tweet(new_tweet):
    new_tweet = new_tweet.replace("<br/>", " ")  # Remplace "<br/>" par un espace
    new_tweet = re.sub(r'https?://[^\s]+', ' ', new_tweet)  # Supprime les URLs
    new_tweet = re.sub(r'[^؀-ۿ]+', ' ', new_tweet)  # Supprime les caractères non-arabes
    new_tweet = re.sub(r'\W+', ' ', new_tweet)  # Supprime les caractères non-alphanumériques
    new_tweet = ' '.join([w for w in new_tweet.split() if len(w) > 1])  # Supprime les mots d'une seule lettre
    return new_tweet

# Fonction pour le traitement avancé du new_tweet
def process(new_tweet):
    new_tweet = araby.strip_tashkeel(new_tweet)  # Supprime les tashkeel
    new_tweet = re.sub(r'\ـ+', ' ', new_tweet)  # Supprime la lettre madda
    new_tweet = re.sub(r'\ر+', 'ر', new_tweet)  # Supprime le ra2 dupliqué
    new_tweet = re.sub(r'\اا+', 'ا', new_tweet)  # Supprime l'alif dupliqué
    new_tweet = re.sub(r'\ووو+', 'و', new_tweet)  # Supprime le waw (plus de 3 fois va à 1)
    new_tweet = re.sub(r'\ههه+', 'ههه', new_tweet)  # Supprime le ha2 (plus de 3 fois va à 1)
    new_tweet = re.sub(r'\ةة+', 'ة', new_tweet)  # Supprime le ta marbuta (plus de 2 fois va à 1)
    new_tweet = re.sub(r'\ييي+', 'ي', new_tweet)  # Supprime le ya (plus de 3 fois va à 1)
    new_tweet = re.sub('أ', 'ا', new_tweet)  # Normalise les variantes de alif
    new_tweet = re.sub('آ', 'ا', new_tweet)  # Normalise les variantes de alif
    new_tweet = re.sub('إ', 'ا', new_tweet)  # Normalise les variantes de alif
    new_tweet = re.sub('ى', 'ي', new_tweet)  # Normalise les variantes de ya
    new_tweet = " ".join(new_tweet.split())  # Supprime les espaces multiples
    return new_tweet

def data_final(new_tweet):
    # Prétraitement du nouveau new_tweet
    new_tweet = remove_nltk_stopwords(new_tweet)
    new_tweet = clean_tweet(new_tweet)
    preprocessed_tweet = process(new_tweet)
    vectorizer = joblib.load("vectorizer.joblib")
    # Vectorisation du new_tweet prétraité
    tweet_vectorized = vectorizer.transform([preprocessed_tweet]).toarray()
    return tweet_vectorized
