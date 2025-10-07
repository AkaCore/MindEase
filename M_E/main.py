import speech_recognition as sr
import pyttsx3
import datetime
import os
import webbrowser
import requests
import wikipedia
import cv2
import numpy as np
import json

# ======================= Initialization =========================
try:
    # This is the primary fix for the -2147483638 error.
    # We explicitly specify the SAPI5 driver for Windows Text-to-Speech.
    engine = pyttsx3.init(driverName='sapi5')
    recognizer = sr.Recognizer()

    # Configure voice properties
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id)  # 0 for male, 1 for female
    else:
        print("Female voice not found, using the default voice.")
    engine.setProperty('rate', 185)
    engine.setProperty('volume', 1.0)

except Exception as e:
    print(f"Failed to initialize speech engine: {e}")
    print("Please ensure your system's text-to-speech service is working.")
    exit()

def speak(text):
    """Function to speak the given text."""
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error during speech: {e}")

# ======================= Startup Menu & Dataset ========================
def startup_menu():
    """Displays an improved menu for user interaction."""
    while True:
        print("\n" + "="*20)
        print("   ASSISTANT MENU")
        print("="*20)
        print("1. Register a new face")
        print("2. Train the face recognition model")
        print("3. Start the assistant")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            create_dataset()
        elif choice == '2':
            train_model()
        elif choice == '3':
            if not os.path.exists("trainer.yml") or not os.path.exists("labels.json"):
                print("\n[ERROR] Model not trained yet! Please select option 2 to train the model first.")
                speak("The recognition model has not been trained yet. Please train it first.")
            else:
                face_recognition_trigger()
        elif choice == '4':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

def create_dataset():
    """Captures and saves face images for a new user."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    person_name = input("Enter the name of the person: ").strip()
    if not person_name:
        print("Name cannot be empty.")
        return
        
    person_path = os.path.join("faces_dataset", person_name)
    os.makedirs(person_path, exist_ok=True)

    cam = cv2.VideoCapture(0)
    count = 0
    print("Look at the camera. Capturing 50 images...")
    
    while count < 50:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(person_path, f"face_{count}.jpg"), face)
            count += 1
            cv2.putText(frame, f"Images: {count}/50", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Capturing Images", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cam.release()
    cv2.destroyAllWindows()
    print(f"Dataset created for {person_name}. Please train the model now (Option 2).")
    speak(f"Dataset created for {person_name}. Please train the model now.")

# ======================= Model Training Function =====================
def train_model():
    """Trains the model and saves it to a file."""
    dataset_path = os.path.join(os.getcwd(), "faces_dataset")
    if not any(os.scandir(dataset_path)):
        print("Dataset is empty. Please register a face first.")
        speak("Dataset is empty. Please register a face first.")
        return

    print("Training model... This might take a moment.")
    speak("Training model. Please wait.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images, labels, names = [], [], {}
    current_id = 0

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path): continue
        
        names[current_id] = person_name
        
        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(current_id)
        current_id += 1
    
    with open("labels.json", 'w') as f:
        json.dump(names, f)

    recognizer.train(images, np.array(labels))
    recognizer.save("trainer.yml")

    print(f"Model trained successfully and saved as trainer.yml")
    speak("Model trained successfully.")

# ======================= Face Recognition Logic =====================
def face_recognition_trigger():
    """Loads the pre-trained model and starts recognition."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    with open("labels.json", 'r') as f:
        names = json.load(f)
        names = {int(k): v for k, v in names.items()}

    speak("Looking for a face to start.")
    recognized, name = recognize_face_from_cam(face_cascade, recognizer, names)
    
    if recognized:
        speak(f"Welcome, {name}. How can I assist you today?")
        listen_for_commands()
    else:
        speak("Face not recognized. Returning to menu.")
        print("Face not recognized.")

def recognize_face_from_cam(face_cascade, recognizer, names):
    """Recognizes a face using the camera and the loaded model."""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not access the camera.")
        return False, None

    print("Looking for a recognized face...")
    
    recognized_name = None
    for _ in range(150): # Look for ~5 seconds
        ret, frame = cam.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            label_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 75:
                recognized_name = names.get(label_id, "Unknown")
                cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                break 

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or recognized_name:
            break
            
    cam.release()
    cv2.destroyAllWindows()
    return (recognized_name is not None), recognized_name

# ======================= Assistant Commands =====================
def get_news():
    # IMPORTANT: Replace with your own key from newsapi.org
    api_key = "YOUR_NEWS_API_KEY_HERE"
    url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        if articles:
            speak("Here are the top headlines from India:")
            for article in articles[:3]:
                print(f"  - {article['title']}")
                speak(article['title'])
        else:
            speak("No news articles were found.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        speak("Sorry, I couldn't fetch the news right now.")

def get_wikipedia_info(topic):
    try:
        speak(f"Searching Wikipedia for {topic}")
        summary = wikipedia.summary(topic, sentences=3, auto_suggest=False)
        print(f"Wikipedia Summary for {topic}:\n{summary}")
        speak("According to Wikipedia:")
        speak(summary)
    except wikipedia.exceptions.PageError:
        speak(f"Sorry, I could not find a Wikipedia page for {topic}.")
    except wikipedia.exceptions.DisambiguationError:
        speak(f"{topic} is ambiguous. Please be more specific.")
    except Exception as e:
        print(f"An error occurred with Wikipedia: {e}")
        speak("Sorry, an error occurred while searching Wikipedia.")

def open_application(command):
    """Handles various voice commands."""
    command = command.lower()
    if 'time' in command:
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The current time is {current_time}")
    elif 'date' in command:
        current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
        speak(f"Today's date is {current_date}")
    elif 'play a song' in command:
        speak("Playing a popular song on YouTube.")
        webbrowser.open("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    elif 'open youtube' in command:
        speak("Opening YouTube")
        webbrowser.open("https://www.youtube.com")
    elif 'open google' in command:
        speak("Opening Google")
        webbrowser.open("https://www.google.com")
    elif 'news' in command:
        get_news()
    elif 'information on' in command or 'tell me about' in command:
        topic = command.replace('information on', '').replace('tell me about', '').strip()
        get_wikipedia_info(topic)
    elif 'exit' in command or 'goodbye' in command or 'shut down' in command:
        speak("Goodbye!")
        return True  # Signal to stop listening
    else:
        speak("Command not recognized. Please try again.")
    return False # Signal to continue listening

def listen_for_commands():
    """Listens for voice commands until the exit command is given."""
    print("\nListening for commands... (e.g., 'what's the time?', 'news', 'exit')")
    while True:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            command = recognizer.recognize_google(audio)
            print(f"You said: {command}")
            if open_application(command):
                break # Exit loop if open_application returns True
        except sr.WaitTimeoutError:
            speak("I didn't hear anything. Returning to menu.")
            break
        except sr.UnknownValueError:
            pass # Ignore if speech is not understood, and just listen again
        except sr.RequestError as e:
            speak("Sorry, my speech service is down.")
            print(f"Could not request results from Google Speech Recognition service; {e}")
            break

# ======================= Main Execution Block =====================
if __name__ == "__main__":
    # Ensure the main dataset folder exists on startup
    os.makedirs("faces_dataset", exist_ok=True)
    speak("Welcome to your personal voice assistant.")
    startup_menu()
