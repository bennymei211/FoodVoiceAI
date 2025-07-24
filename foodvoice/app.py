from openai import OpenAI
from dotenv import load_dotenv
import pyttsx3
import time
import os
import asyncio
import edge_tts

load_dotenv()

# initialize OpenAI object with API Key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_gpt_response(user_input):
    message = [
        {
            "role": "developer",
            "content": "Gather information on what I had for a certain meal. Such as nutritional value, keeping a log of what I have, etc. Make sure to ask for each meal. Respond in a way that is text-to-speech friendly"
        },
        {
            "role": "assistant",
            "content": "Hello! What did you eat for breakfast?"
        },
        {
            "role": "user",
            "content": user_input
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=message
    )
    return response.choices[0].message.content


def speak(text, engine):
    engine.say(text)
    engine.runAndWait()


if __name__ == "__main__":
    engine = pyttsx3.init()
    print("ChatBot: Hello! What did you have for breakfast?")
    speak("Hello! What did you have for breakfast?", engine)
    while True:
        engine.say("...")
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            speak("Thanks for sharing! Goodbye!", engine)
            print("ChatBot: Thanks for sharing! Goodbye!")
            time.sleep(4)
            break
        response = get_gpt_response(user_input)
        speak(response, engine)
        print(f"Chatbot: {response}")
    engine.stop()

