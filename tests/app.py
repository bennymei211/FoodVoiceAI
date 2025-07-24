import time
import speech_recognition as sr
import pyttsx3


conversation = []
meals = {
    "breakfast": "",
    "lunch": "",
    "dinner": "",
    "dessert": ""
}


def speak(text, engine):
    engine.say(text)
    engine.runAndWait()


def respond_to_user(user_input, engine):
    user_input = user_input.lower()
    if "breakfast" in user_input:
        meals["breakfast"] = user_input
        speak("Yum! What did you have for lunch?", engine)
        return "Yum! What did you have for lunch?"
    elif "lunch" in user_input:
        meals["lunch"] = user_input
        speak("Nice. Did you have anything for dinner?", engine)
        return "Nice. Did you have anything for dinner?"
    elif "dinner" in user_input:
        meals["dinner"] = user_input
        speak("Sounds tasty! Did you have dessert?", engine)
        return "Sounds tasty! Did you have dessert?"
    elif "dessert" in user_input:
        meals["dessert"] = user_input
    elif "summary" or "remind" in user_input:
        summary = "Here's what you've had so far:\n"
        for meal, details in meals.items():
            if details:
                summary += f"{meal.title()}: {details}\n"
            else:
                summary += f"{meal.title()}: Not shared yet.\n"
        speak(summary, engine)
        time.sleep(10)
        return summary
    else:
        speak("Thanks for sharing! Did you eat anything else?", engine)
        return "Thanks for sharing! Did you eat anything else?"


def main():
    engine = pyttsx3.init()
    recognizer = sr.Recognizer()
    print("ChatBot: Hello! What did you eat today?")
    speak("Hello! What did you eat today?", engine)
    while True:
        try:
            engine.say("...")
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
                user = recognizer.recognize_google(audio)
                conversation.append(user)
                print("You:", user)
                if user.lower() in ["quit", "exit"]:
                    speak("Thanks for sharing! Goodbye!", engine)
                    print("ChatBot: Thanks for sharing! Goodbye!")
                    time.sleep(4)
                    engine.stop()
                    return
                reply = respond_to_user(user, engine)
                print("ChatBot:", reply)
        except:
            time.sleep(2)
            print("Sorry, I couldn't understand.")
            speak("Sorry, I couldn't understand", engine)


main()
