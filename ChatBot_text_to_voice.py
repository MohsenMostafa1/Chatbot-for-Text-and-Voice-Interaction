from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import pyttsx3

# Load the BlenderBot model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Initialize the text-to-speech engine
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

print("Chatbot is ready! Type 'exit' to stop the conversation.")
while True:
    user_input = input("You: ")  # Change from listen() to input()
    if user_input.lower() == 'exit':
        break
    bot_response = chat_with_bot(user_input)
    print("Bot:", bot_response)
    speak(bot_response)
