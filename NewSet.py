import os
from ultralytics import YOLO
import cv2
import requests
import json
import logging

VIDEOS_DIR = os.path.join('.', 'videos')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

ret, frame = cap.read()
H, W, _ = frame.shape

out = cv2.VideoWriter('webcam_out.mp4', cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'best.pt')

# Load a model
model = YOLO(model_path)

threshold = 0.5

SERVER_URL = 'http://127.0.0.1:5000/update_stock'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_stock(product_id, quantity):
    data = {'product_id': product_id, 'quantity': quantity}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(SERVER_URL, data=json.dumps(data), headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        logger.info(f"Server response: {response.json()}")
        return True  # Indicate success
    except requests.exceptions.RequestException as e:
        logger.error(f"Error updating stock: {e}")
        return False  # Indicate failure


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            class_name = model.names[int(class_id)]
            cv2.putText(frame, class_name.upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            detected_product_id = str(int(class_id))
            detected_quantity = 1

            # Call the update_stock function to notify the server
            success = update_stock(detected_product_id, detected_quantity)

            # Visualize success or failure
            color = (0, 255, 0) if success else (0, 0, 255)
            cv2.putText(frame, "Updated" if success else "Update Failed", (int(x1), int(y2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    cv2.imshow('Webcam Detection', frame)
    out.write(frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
