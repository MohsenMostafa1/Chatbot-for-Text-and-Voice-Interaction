from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from gtts import gTTS
import os
import playsound

# Load the BlenderBot model and tokenizer
# you can use large model like model_name = "facebook/blenderbot-3B" 
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

def chat_with_bot(user_input):
    # Encode the input and generate a response
    inputs = tokenizer(user_input, return_tensors='pt')
    reply_ids = model.generate(**inputs)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return response

def speak(text):
    # Use gTTS to convert text to speech and play it
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    playsound.playsound("response.mp3")
    os.remove("response.mp3")  # Remove the audio file after playing

print("Chatbot is ready! Type 'exit' to stop the conversation.")
while True:
    user_input = input("You: ")  # Change from listen() to input()
    if user_input.lower() == 'exit':
        break
    bot_response = chat_with_bot(user_input)
    print("Bot:", bot_response)
    speak(bot_response)
