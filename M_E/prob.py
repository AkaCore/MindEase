
import speech_recognition as sr
import pyttsx3
import datetime
import os
import webbrowser
import requests
import wikipedia
import cv2
import numpy as np

# Initialize recognizer and text-to-speech engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Voice Adjustment
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # 0 for male, 1 for female
engine.setProperty('rate', 185)  # Speed (default is 200)
engine.setProperty('volume', 1.0)

# Authorizing face OR New entry of face 
def create_dataset():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    person_name = input("Enter the name of the person: ")
    person_path = os.path.join("faces_dataset", person_name)
    os.makedirs(person_path, exist_ok=True)

    cam = cv2.VideoCapture(0)
    count = 0

    while count < 50:  # Adjust the number of images you want to capture
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image. Try again.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            img_path = os.path.join(person_path, f"face_{count}.jpg")
            cv2.imwrite(img_path, face)
            count += 1

        cv2.imshow("Capturing Images", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Dataset created for {person_name} with {count} images.")

# Face Recognition Functions
def capture_faces():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    dataset_path = r"D:\Study\Py_L_Vs\faces_dataset"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")
    
    if not os.listdir(dataset_path):
        raise ValueError("The dataset folder is empty. Please add labeled subfolders with images.")

    images, labels = [], []

    # Load the dataset
    for label, person_name in enumerate(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person_name)
        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            labels.append(label)

    recognizer.train(images, np.array(labels))
    return face_cascade, recognizer, os.listdir(dataset_path)
#face reco
def recognize_face(face_cascade, recognizer, labels):
    cam = cv2.VideoCapture(1)
    print("Looking for your face...")

    if not cam.isOpened():
        print("Error: Could not access the camera.")
        return False

    print("Looking for your face...")

    while True:
        ret, frame = cam.read()


        if not ret:
            print("Failed to grab frame!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print("No faces detected!")
        else:
            print(f"Detected faces: {len(faces)}")  # Debugging line

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face)

            print(f"Predicted label: {label}, Confidence: {confidence}")  # Debugging line

            # Adjust the confidence threshold if necessary
            if confidence < 70:
                print(f"Face recognized as {labels[label]} with confidence {confidence}")
                cam.release()
                cv2.destroyAllWindows()
                speak(f"Welcome, {labels[label]}")
                return True

    cam.release()
    cv2.destroyAllWindows()
    return False


# Replace wake word detection with face recognition
def face_recognition_trigger():
    face_cascade, recognizer, labels = capture_faces()
    while True:
        if recognize_face(face_cascade, recognizer, labels):
            speak("How can I assist you today?")
            if listen_for_commands():
                break

# Function to get the current time
def get_time():
    now = datetime.datetime.now()
    return now.strftime("%H:%M")

# Function to get the current date
def get_date():
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y")

# Function to handle commands
def open_application(command):
    print(f"Processing command: {command}")  
    if 'time' in command.lower():
        current_time = get_time()
        speak(f"The current time is {current_time}")
    elif 'date' in command.lower():
        current_date = get_date()
        speak(f"Today's date is {current_date}")

    elif 'play song' in command.lower():
        speak("Playing")
        webbrowser.open("https://www.youtube.com/watch?v=EsTx07oNQNQ&list=LL")

    elif 'open youtube' in command.lower():
        speak("Opening YouTube")
        webbrowser.open("https://www.youtube.com")

    elif 'open chrome' in command.lower():
        speak("Opening Chrome")
        os.system("start chrome")

    elif 'open facebook' in command.lower():
        speak("Opening Facebook")
        webbrowser.open("https://www.facebook.com")

    elif 'open steam' in command.lower():
        speak("Opening Steam")
        os.system(r'"C:\Program Files (x86)\Steam\Steam.exe"')

    elif 'shut down' in command.lower():  
        speak("Shutting down the system...")
        os.system('shutdown /s /f /t 0')

    elif 'get news' in command.lower():
        get_news()

    elif 'google' in command.lower():
        speak("Opening Google")
        webbrowser.open("https://www.google.com/")

    # To open the website on basis of website name
    elif 'open' in command.lower():
        speak("Which website would you like to open?")
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            try:
                website_name = recognizer.recognize_google(audio).lower().strip()
                constructed_url = f"https://{website_name}.com"
                print(constructed_url)
                speak(f"Opening {website_name}")
                webbrowser.open(constructed_url)
            except sr.UnknownValueError:
                speak("Please say the website name clearly.")
            except sr.RequestError:
                speak("There was an issue with the recognition service.")

    # function to get info from wiki
    elif 'get information' in command.lower():
        speak("What topic information would you like to search sir? ")
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            try:
                topic = recognizer.recognize_google(audio).strip()
                speak(f"Searching Wikipedia for {topic}")

                # Attempt to fetch the summary from Wikipedia
                try:
                    summary = wikipedia.summary(topic, sentences=5)  # Fetch 5 sentences
                    speak(f"Here is what I found about {topic}:")
                    speak(summary)

                    # Save to a txt file
                    with open(f"{topic}.txt", "w", encoding="utf-8") as file:
                        file.write(f"Wikipedia Summary for {topic}:\n\n{summary}")
                    speak(f"The information has been saved to {topic}.txt")
                except wikipedia.exceptions.DisambiguationError:
                    speak("The topic is ambiguous. Please be more specific.")
                except wikipedia.exceptions.PageError:
                    speak("Sorry, I couldn't find any information on that topic.")
                except wikipedia.exceptions.HTTPTimeoutError:
                    speak("There was a timeout while fetching the Wikipedia page. Please try again later.")
                except Exception as e:
                    speak(f"An error occurred while fetching the Wikipedia page: {e}")
            except sr.UnknownValueError:
                speak("I couldn't understand the topic. Please say it clearly.")
            except sr.RequestError:
                speak("There was an issue with the recognition service.")
            except Exception as e:
                speak(f"An error occurred with speech recognition: {e}")

    # To stop the listening loop
    elif 'exit' in command.lower():
        speak("See you soon sir")
        return True  

    else:
        speak("Command not recognized. Please try again.")
        return False

# News Section
def get_news():
    api_key = "ce495cd379e8429a82925e35d8cff755"  
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        articles = data['articles']
        if articles:
            speak(f"Here are the latest news headlines:")
            for article in articles[:5]:  
                title = article['title']
                speak(f"{title}")
        else:
            speak("No news articles available at the moment.")
    else:
        speak("Sorry, I couldn't fetch the news at the moment.")



# Google ap

# Function to listen for commands
def listen_for_commands():
    print("Now listening for commands...")
    while True:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening for commands...")
            audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio)
            print(f"You said: {command}")
            if open_application(command):
                return True
            else:
                continue
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            speak("Sorry, the service is down.")

# Start face recognition trigger
face_recognition_trigger()
