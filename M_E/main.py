import speech_recognition as sr
import pyttsx3
import datetime
import os
import webbrowser
import requests
import wikipedia
import cv2
import numpy as np

# ======================= Initialization =========================
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Configure voice properties
voices = engine.getProperty('voices')
if len(voices) > 1:
    engine.setProperty('voice', voices[1].id)  # 0 for male, 1 for female
else:
    print("Female voice not found, using default.")
engine.setProperty('rate', 185)
engine.setProperty('volume', 1.0)

def speak(text):
    """Function to speak the given text."""
    engine.say(text)
    engine.runAndWait()

# ======================= Startup Menu & Dataset ========================
def startup_menu():
    """Displays a menu for the user to register faces or start recognition."""
    dataset_folder = os.path.join(os.getcwd(), "faces_dataset")
    os.makedirs(dataset_folder, exist_ok=True)
    
    while True:
        registered_faces = [name for name in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, name))]
        print(f"\n--- Assistant Menu ---")
        print(f"{len(registered_faces)} face(s) registered.")
        print("1. Register a new face")
        print("2. Start face recognition")
        print("3. Exit")
        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == '1':
            create_dataset()
        elif choice == '2':
            if not registered_faces:
                print("No faces registered yet! Please register at least one face t1o continue.")
                speak("No faces registered yet! Please register at least one face to continue.")
            else:
                return True # Proceed to face recognition
        elif choice == '3':
            return False # Exit the program
        else:
            print("Invalid choice. Please try again.")

def create_dataset():
    """Captures and saves face images for a new user."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    person_name = input("Enter the name of the person: ").strip()
    person_path = os.path.join("faces_dataset", person_name)
    os.makedirs(person_path, exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return
        
    count = 0
    print("Look at the camera. Capturing 50 images...")

    while count < 50:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image. Exiting.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            img_path = os.path.join(person_path, f"face_{count}.jpg")
            cv2.imwrite(img_path, face)
            count += 1
            # Display the progress on the frame
            cv2.putText(frame, f"Images Captured: {count}/50", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        cv2.imshow("Capturing Images", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Dataset created for {person_name} with {count} images.")
    speak(f"Dataset created for {person_name}.")

# ======================= Face Recognition Logic =====================
def load_and_train_recognizer():
    """Loads images from the dataset and trains the face recognizer."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    dataset_path = os.path.join(os.getcwd(), "faces_dataset")

    images, labels, names = [], [], {}
    current_id = 0

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        names[current_id] = person_name
        
        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(current_id)
        
        current_id += 1

    if not images:
         raise ValueError("The dataset folder is empty or contains no valid images.")

    recognizer.train(images, np.array(labels))
    return face_cascade, recognizer, names

def recognize_face(face_cascade, recognizer, names):
    """Attempts to recognize a face from the camera feed."""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not access the camera.")
        return False, None

    print("Looking for a recognized face...")
    
    recognized = False
    recognized_name = None

    for _ in range(150): # Try for about 5 seconds
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            label_id, confidence = recognizer.predict(face)

            # Lower confidence value means better match
            if confidence < 75:
                recognized_name = names[label_id]
                cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                recognized = True
            else:
                 cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or recognized:
            break

    cam.release()
    cv2.destroyAllWindows()
    return recognized, recognized_name

def face_recognition_trigger():
    """Main loop for face recognition and command listening."""
    try:
        face_cascade, recognizer, names = load_and_train_recognizer()
        recognized, name = recognize_face(face_cascade, recognizer, names)
        
        if recognized:
            speak(f"Welcome, {name}. How can I assist you today?")
            listen_for_commands()
        else:
            speak("Face not recognized. Shutting down.")
            print("Face not recognized.")
    except ValueError as e:
        print(e)
        speak(str(e))


# ======================= Assistant Commands =====================
def get_time():
    return datetime.datetime.now().strftime("%H:%M")

def get_date():
    return datetime.datetime.now().strftime("%A, %B %d, %Y")

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
            speak("Here are the top news headlines:")
            for article in articles[:3]: # Read top 3
                speak(article['title'])
        else:
            speak("No news articles available at the moment.")
    except requests.exceptions.RequestException:
        speak("Sorry, I couldn't fetch the news at the moment.")

def open_application(command):
    """Handles various voice commands."""
    command = command.lower()
    if 'time' in command:
        speak(f"The current time is {get_time()}")
    elif 'date' in command:
        speak(f"Today's date is {get_date()}")
    elif 'play song' in command:
        speak("Playing a song on YouTube.")
        webbrowser.open("https://www.youtube.com/watch?v=dQw4w9WgXcQ") # A classic
    elif 'open youtube' in command:
        speak("Opening YouTube")
        webbrowser.open("https://www.youtube.com")
    elif 'open chrome' in command:
        speak("Opening Chrome")
        os.system("start chrome")
    elif 'get news' in command:
        get_news()
    elif 'google' in command:
        speak("Opening Google")
        webbrowser.open("https://www.google.com")
    elif 'get information' in command:
        speak("What topic would you like to know about?")
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
                topic = recognizer.recognize_google(audio).strip()
                speak(f"Searching Wikipedia for {topic}")
                summary = wikipedia.summary(topic, sentences=3)
                speak("According to Wikipedia:")
                speak(summary)
        except sr.UnknownValueError:
            speak("I couldn't understand the topic.")
        except (sr.RequestError, wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError) as e:
            speak(f"Sorry, I could not find information on that topic. Error: {e}")
    elif 'exit' in command or 'goodbye' in command:
        speak("Goodbye!")
        return True # Signal to stop listening
    else:
        speak("Command not recognized.")
    return False # Signal to continue listening

def listen_for_commands():
    """Listens for voice commands until the exit command is given."""
    print("Listening for commands...")
    while True:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            command = recognizer.recognize_google(audio)
            print(f"You said: {command}")
            if open_application(command):
                break
        except sr.UnknownValueError:
            pass # Ignore if speech is not understood
        except sr.RequestError:
            speak("Sorry, the recognition service is down.")
        except sr.WaitTimeoutError:
            speak("I didn't hear anything. Shutting down.")
            break


# ======================= Main Execution Block =====================
if __name__ == "__main__":
    # The startup_menu function will return True if the user wants to proceed
    # with face recognition, and False if they choose to exit.
    if startup_menu():
        face_recognition_trigger()
    
    print("Program finished.")
