import pickle
import os
import re
from datetime import datetime

MODEL_PATH = "model/emotion_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Model not found!")
    print("Please run: python train.py")
    exit()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def detect_emotion():
    text = input("\nEnter your text: ").strip()

    if not text:
        print(" Text cannot be empty.")
        return
    cleaned_text = clean_text(text)
    prediction = model.predict([cleaned_text])[0]
    probabilities = model.predict_proba([cleaned_text])[0]
    confidence = max(probabilities) * 100

    emoji_map = {
        "joy": "😊",
        "sadness": "😢",
        "anger": "😡",
        "fear": "😨",
        "love": "❤️",
        "surprise": "😲"
    }

    emoji = emoji_map.get(prediction, "")
    print(f" Predicted Emotion: {prediction.upper()} {emoji}")
    print(f" Confidence: {confidence:.2f}%")

    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("history.txt", "a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] Text: {text} | Emotion: {prediction} | Confidence: {confidence:.2f}%\n")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def view_history():
    if not os.path.exists("history.txt"):
        print(" No history found.")
        return

    print("\n Prediction History:\n")
    with open("history.txt", "r", encoding="utf-8") as file:
        print(file.read())

def clear_history():
    if os.path.exists("history.txt"):
        open("history.txt", "w", encoding="utf-8").close()
        print("History cleared successfully.")
    else:
        print(" No history file found.")


def show_help():
    print("""
Available Options:
1. Detect Emotion - Enter text and predict emotion
2. View History - Show previous predictions
3. Clear History - Delete saved prediction history
4. Help - Show this help menu
5. Exit - Close the application
""")

def main():
    while True:
        print("\n" + "="*40)
        print("Emotion Detection CLI App")
        print("="*40)
        print("1. Detect Emotion")
        print("2. View History")
        print("3. Clear History")
        print("4. Help")
        print("5. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            detect_emotion()
        elif choice == "2":
            view_history()
        elif choice == "3":
            clear_history()
        elif choice == "4":
            show_help()
        elif choice == "5":
            print("Exiting application. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()
