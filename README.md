# Chatbot-for-Text-and-Voice-Interaction

#### In today's digital landscape, chatbots have emerged as pivotal tools for enhancing user interaction across various platforms. These intelligent systems can engage users through text or voice, providing a seamless experience that caters to individual preferences. 

<figure>
        <img src="https://floatbot.ai/img/Voicebot-Vs-Chatbot.png" alt ="Audio Art" style='width:800px;height:500px;'>
        <figcaption>

### installing necessary Libraries 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install playsound==1.2.2

pip install pygame

pip install gtts

## Explaination for appChat_Text_To_Voice

transformers: This library is used to load pre-trained models and tokenizers for natural language processing tasks. 
Here, BlenderbotTokenizer and BlenderbotForConditionalGeneration are imported to handle the tokenization and model generation tasks.
gtts: The Google Text-to-Speech library is imported to convert text responses from the chatbot into speech.
os: This module provides a way of using operating system-dependent functionality like reading or writing to the file system.
pygame: A library for creating games and multimedia applications, used here for playing audio.

```python
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from gtts import gTTS  # Make sure to import gTTS
import os
import pygame
```

The model name "facebook/blenderbot-400M-distill" specifies which pre-trained model to load. 
This model is a distilled version of BlenderBot, which is smaller yet effective for conversational tasks.
The tokenizer converts input text into a format that the model can process (token IDs), while the model itself generates responses based on these tokens.

```python
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
```
chat_with_bot: This function takes user input as a string.

It tokenizes the input using the tokenizer and converts it into PyTorch tensors.
The model generates a response based on these inputs.
Finally, it decodes the generated token IDs back into human-readable text, skipping any special tokens.

```python
def chat_with_bot(user_input):
    inputs = tokenizer(user_input, return_tensors='pt')
    reply_ids = model.generate(**inputs)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return response
```

speak: This function converts text to speech.
It uses gTTS to create an audio file (response.mp3) from the text.
Pygame initializes its mixer module to play audio.
The function waits until the audio finishes playing before cleaning up by quitting Pygame and removing the temporary audio file.

```python
def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = 'response.mp3'
    tts.save(filename)
    
    try:
        pygame.init()
        pygame.mixer.init()
        
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Check every 100ms
        
    finally:
        pygame.quit()
        os.remove(filename)  # Remove temporary file after playback
```

The main loop continuously prompts the user for input until they type "exit".
For each user input, it calls chat_with_bot to get a response from the chatbot and then prints this response.
Finally, it calls speak to convert the response into speech and play it.

```python
print("Chatbot is ready! Type 'exit' to stop the conversation.")
while True:
    user_input = input("You: ")  

    if user_input.lower() == 'exit':
        break

    bot_response = chat_with_bot(user_input)
    
    print(f"Bot: {bot_response}")
    
    speak(bot_response)
```






