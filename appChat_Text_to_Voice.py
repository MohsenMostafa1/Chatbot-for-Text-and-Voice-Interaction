from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from gtts import gTTS  # Make sure to import gTTS
import os
import pygame

# Load the BlenderBot model and tokenizer
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
    # Use gTTS to convert text to speech and play it with Pygame
    tts = gTTS(text=text, lang='en')
    
    # Save audio file temporarily before playing it with Pygame
    filename = 'response.mp3'
    tts.save(filename)
    
    try:
        pygame.init()
        pygame.mixer.init()
        
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        
        # Wait until the music finishes playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Check every 100ms
        
    finally:
        pygame.quit()
        os.remove(filename)  # Remove temporary file after playback

print("Chatbot is ready! Type 'exit' to stop the conversation.")
while True:
    user_input = input("You: ")  

    if user_input.lower() == 'exit':
        break

    bot_response = chat_with_bot(user_input)
    
    print(f"Bot: {bot_response}")
    
    speak(bot_response)
