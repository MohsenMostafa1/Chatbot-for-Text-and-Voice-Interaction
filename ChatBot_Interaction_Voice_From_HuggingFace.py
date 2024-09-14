import pyttsx3
import speech_recognition as sr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load a pre-trained model and tokenizer
model_name = "CAMeL-Lab/bert-base-arabic-camelbert-ca"  # Use any appropriate LLM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a pipeline for conversational chat
chatbot = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to listen for voice input
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio)
            print(f"You said: {user_input}")
            return user_input
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return ""
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return ""

# Start the chatbot loop
def chatbot_conversation():
    speak("Hello! How can I assist you today?")
    
    while True:
        # Take user input via voice
        user_input = listen()
        
        # Break loop if user types 'exit' or 'quit'
        if user_input.lower() in ['exit', 'quit']:
            speak("Goodbye!")
            break
        
        # Generate a response using the chatbot model
        response = chatbot(user_input, max_length=100, num_return_sequences=1)
        
        # Print and speak the chatbot's response
        chatbot_response = response[0]['generated_text']
        print(f"Chatbot: {chatbot_response}")
        speak(chatbot_response)

# Run the chatbot
if __name__ == "__main__":
    chatbot_conversation()
