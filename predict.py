
from ultralytics import YOLO
import os
current_dir = os.getcwd()
images_dir = os.path.join("images","grapes.jpeg")
model_dir=os.path.join("runs","detect","train","weights","best.pt")
image_path=os.path.join(current_dir,images_dir)
model_path=os.path.join(current_dir,model_dir)
print(model_path)
# Load the model with the best weights
model=YOLO(model_path)
# Make predictions
results = model.predict(image_path,conf=0.7)
detected_objects = set()
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        detected_objects.add(result.names[class_id])

print(list(detected_objects))