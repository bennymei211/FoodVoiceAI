import speech_recognition as sr
import pyttsx3


def speak(text, engine):
    print("ChatBot:", text)  # For debugging/logging
    engine.say(text)
    engine.runAndWait()


def respond_to_user(user_input):
    user_input = user_input.lower()
    if "breakfast" in user_input:
        return "Yum! What did you have for lunch?"
    elif "lunch" in user_input:
        return "Nice. Did you have anything for dinner?"
    elif "dinner" in user_input:
        return "Sounds tasty! Did you have dessert?"
    else:
        return "Thanks for sharing! Did you eat anything else?"


def main():
    engine = pyttsx3.init()
    recognizer = sr.Recognizer()

    greeting = "Hello! What did you eat today?"
    print("ChatBot:", greeting)
    speak(greeting, engine)

    while True:
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)

        try:
            user = recognizer.recognize_google(audio)
            print("You:", user)

            if user.lower() in ["quit", "exit"]:
                farewell = "Thanks for sharing! Goodbye!"
                speak(farewell, engine)
                print("ChatBot:", farewell)
                engine.stop()
                break

            reply = respond_to_user(user)
            speak(reply, engine)
        except Exception as e:
            print("Error:", e)
            error_msg = "Sorry, I couldn't understand."
            speak(error_msg, engine)

main()
