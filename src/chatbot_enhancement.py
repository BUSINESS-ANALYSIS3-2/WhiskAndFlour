import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Sample training data: (user input, intent)
training_data = [
    ("I want something vegan", "vegan_request"),
    ("Do you have gluten-free bread?", "gluten_free_request"),
    ("Tell me about croissants", "product_info"),
    ("I’d like to order a birthday cake", "place_order"),
    ("What do you recommend today?", "recommendation"),
    ("Can I get a cinnamon roll?", "place_order"),
    ("Do you have sugar-free options?", "diet_request"),
    ("What’s popular this week?", "recommendation"),
    ("I want to order something sweet", "place_order"),
    ("Tell me about your baguettes", "product_info"),
]

# Preprocessing
stop_words = set(stopwords.words("english"))
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join([t for t in tokens if t.isalpha() and t not in stop_words])

# Vectorization
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform([preprocess(text) for text, label in training_data])
y_train = [label for text, label in training_data]

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Response dictionary
responses = {
    "vegan_request": "We have Vegan Chocolate Tart and Vegan Banana Muffins. Would you like to place an order?",
    "gluten_free_request": "Yes! Our gluten-free options include almond flour bread and coconut cookies.",
    "diet_request": "We offer sugar-free lemon bars and keto brownies. Interested?",
    "product_info": "Our croissants and baguettes are freshly baked every morning. Would you like to try one?",
    "place_order": "Great! What would you like to order today?",
    "recommendation": "Our cinnamon rolls and chocolate tarts are trending this week. Want to try one?",
}

# Chatbot function
def chatbot():
    print("👩‍🍳 Welcome to Whisk & Flour Bakery Chatbot!")
    print("Type 'exit' to end the chat.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Bot: Thank you for visiting Whisk & Flour. Have a sweet day!")
            break
        processed = preprocess(user_input)
        vectorized = vectorizer.transform([processed])
        intent = model.predict(vectorized)[0]
        response = responses.get(intent, "I'm here to help! Could you please clarify your request?")
        print("Bot:", response)

# Run chatbot
if __name__ == "__main__":
    chatbot()

