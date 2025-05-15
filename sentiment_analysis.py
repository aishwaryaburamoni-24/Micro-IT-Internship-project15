from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sentences = [
    # Positive
    "I love this product",
    "This is the best thing ever",
    "Amazing quality and service",
    "Im really happy with it",
    "Fast delivery and good experience",
    "It worked perfectly, thanks!",
    "Highly recommend to everyone",
    "Very satisfied with my purchase",

    # Negative
    "This is the worst experience",
    "I hate it so much",
    "Very disappointed and angry",
    "Complete waste of time and money",
    "Not good at all",
    "I will never buy this again",
    "Terrible product and poor quality",
    "The service was awful"
]

labels = [1]*8 + [0]*8 

vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(sentences)
model = LogisticRegression()
model.fit(features, labels)

def predict_sentiment(text):
  vector = vectorizer.transform([text])
  result = model.predict(vector)[0]
  return "Positive" if result == 1 else "Negative"

if __name__ == "__main__":
  print("Sentiment Analyzer (type 'exit' to quit)\n")
  while True:
    sentence = input("Enter a sentence: ")
    if sentence.lower() == "exit":
        break
    print("Sentiment:", predict_sentiment(sentence))
    print()
