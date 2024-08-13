from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import speech_recognition as sr
import pyttsx3

# Load the BlenderBot model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Initialize the speech recognition and text-to-speech engines
recognizer = sr.Recognizer()
engine = pyttsx3.init(driverName='sapi5')  # Use 'sapi5' for Windows TTS

def chat_with_bot(user_input):
    # Encode the input and generate a response
    inputs = tokenizer(user_input, return_tensors='pt')
    reply_ids = model.generate(**inputs)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return response

def speak(text):
    # Use text-to-speech to speak the response
    engine.say(text)
    engine.runAndWait()

def listen():
    # Use the microphone to listen for user input
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio)
            print("You:", user_input)
            return user_input
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return None

print("Chatbot is ready! Say 'exit' to stop the conversation.")
while True:
    user_input = listen()
    if user_input is None:
        continue
    if user_input.lower() == 'exit':
        break
    bot_response = chat_with_bot(user_input)
    print("Bot:", bot_response)
    speak(bot_response)
