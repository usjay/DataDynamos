import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import RPi.GPIO as GPIO
import time
import mysql.connector

# Setup GPIO pins
BLUE_LED_PIN = 17
RED_LED_PIN = 18
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(BLUE_LED_PIN, GPIO.OUT)
GPIO.setup(RED_LED_PIN, GPIO.OUT)

# Load TFLite models and allocate tensors.
identifier_interpreter = tflite.Interpreter(model_path="/home/data/Desktop/identifierNEW.tflite")
identifier_interpreter.allocate_tensors()

eye_state_interpreter = tflite.Interpreter(model_path="/home/data/Desktop/eye_state_cnn_model.tflite")
eye_state_interpreter.allocate_tensors()

# Get input and output tensors.
identifier_input_details = identifier_interpreter.get_input_details()
identifier_output_details = identifier_interpreter.get_output_details()

eye_state_input_details = eye_state_interpreter.get_input_details()
eye_state_output_details = eye_state_interpreter.get_output_details()

# Initialize the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

def get_phone_number(predicted_name):
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="data",  # Your MySQL password
        database="person_info"
    )
    cursor = connection.cursor()
    query = f"SELECT phone_number FROM person_data WHERE name = '{predicted_name}'"
    try:
        cursor.execute(query)
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            return "No data found for this person."
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None
    finally:
        cursor.close()
        connection.close()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Preprocess frame for the identifier model
    identifier_input_shape = identifier_input_details[0]['shape']
    identifier_input_data = cv2.resize(frame, (identifier_input_shape[1], identifier_input_shape[2]))
    identifier_input_data = np.expand_dims(identifier_input_data, axis=0).astype(np.float32)

    # Make prediction with identifier model
    identifier_interpreter.set_tensor(identifier_input_details[0]['index'], identifier_input_data)
    identifier_interpreter.invoke()
    identifier_output_data = identifier_interpreter.get_tensor(identifier_output_details[0]['index'])
    identifier_predicted_label = np.argmax(identifier_output_data, axis=1)

    # List of names corresponding to the model's output classes
    names = ["Udara", "Ashan", "Niruthmi", "Nethmini", "Diwya"]
    person_name = names[identifier_predicted_label[0]]
    phone_number = get_phone_number(person_name)
    print(f"Predicted Person: {person_name}, Phone number: {phone_number}")

    # Preprocess frame for the eye state model
    eye_state_input_shape = eye_state_input_details[0]['shape']
    eye_input_data = cv2.resize(frame, (eye_state_input_shape[1], eye_state_input_shape[2]))
    eye_input_data = cv2.cvtColor(eye_input_data, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    eye_input_data = np.expand_dims(eye_input_data, axis=-1)  # Add channel dimension (1 channel for grayscale)
    eye_input_data = np.expand_dims(eye_input_data, axis=0).astype(np.float32)

    # Make prediction with eye state model
    eye_state_interpreter.set_tensor(eye_state_input_details[0]['index'], eye_input_data)
    eye_state_interpreter.invoke()
    eye_state_output_data = eye_state_interpreter.get_tensor(eye_state_output_details[0]['index'])
    eye_state_predicted_label = np.argmax(eye_state_output_data, axis=1)

    # Control LEDs based on the predictions
    if person_name == "Udara":
        GPIO.output(BLUE_LED_PIN, GPIO.HIGH)
        GPIO.output(RED_LED_PIN, GPIO.LOW)
    else:
        GPIO.output(BLUE_LED_PIN, GPIO.LOW)
        GPIO.output(RED_LED_PIN, GPIO.HIGH)

    # Display the frame with predictions
    cv2.putText(frame, f"Person: {person_name}, Phone: {phone_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Eye State: {eye_state_predicted_label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Camera Frame', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

# Cleanup GPIO
GPIO.cleanup()