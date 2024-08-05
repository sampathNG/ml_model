# import sys
# print(sys.executable)

from fastapi import APIRouter, Depends, FastAPI,HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import shutil
import os
import pytesseract
pytesseract.pytesseract.tesseract_cmd="/usr/bin/tesseract"
from skimage.filters import threshold_local
import numpy as np
import cv2
# import imutils
router = APIRouter()
@router.get("/")
async def hello():
    return {"message": "Hello World"}
@router.post("/predict")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        new_current_dir = os.getcwd()
        parent_dir = os.path.abspath(os.path.join(new_current_dir, os.pardir))
        os.chdir(parent_dir)
        current_dir = os.getcwd()
        file_paths = f"images/{file.filename}"
        with open(file_paths, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_path =os.path.join(current_dir, f"images/{file.filename}")
        model_path=os.path.join(current_dir,"ml_model","runs","detect","train","weights","best.pt")
        model=YOLO(model_path)
        # Make predictions
        results = model.predict(file_path,conf=0.7)
        detected_objects = set()
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                detected_objects.add(result.names[class_id])
        response_data = {
        "success": True,
        "message": "Objects detected successfully",
        "detected_objects": list(detected_objects)
        }
        os.remove(file_path)
        return JSONResponse(response_data, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)})
    
@router.post("/scanDetect")
async def scan_file():
    try:
         # Initialize the camera
        cap = cv2.VideoCapture(0)  # 0 is usually the default camera
        
        # Capture a single frame
        ret, frame = cap.read()
        
        # Release the camera
        cap.release()
        # scaning image
        # image = cv2.imread("image.jpg")
        # ratio = image.shape[0] / 500.0
        # orig = image.copy()
        # image = imutils.resize(image, height = 500)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.bilateralFilter(gray, 11, 17, 17)  # 11  //TODO 11 FRO OFFLINE MAY NEED TO TUNE TO 5 FOR ONLINE
        # gray = cv2.medianBlur(gray, 5)
        # edged = cv2.Canny(gray, 30, 400)
        # 
        current_dir = os.getcwd()
        model_path=os.path.join(current_dir, "ml_model", "runs", "detect", "train", "weights", "best.pt")
        model=YOLO(model_path)
        # Make predictions
        # results = model.predict(edged, conf=0.7)
        results = model.predict(frame, conf=0.7)
        detected_objects = set()
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                detected_objects.add(result.names[class_id])
        response_data = {
        "success": True,
        "message": "Objects detected successfully",
        "detected_objects": list(detected_objects)
        }
        return JSONResponse(response_data, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)})

@router.post("/ocr/{textt}")
async def create_upload_file(textt: str,file: UploadFile = File(...)):
    try:
        new_current_dir = os.getcwd()
        parent_dir = os.path.abspath(os.path.join(new_current_dir, os.pardir))
        os.chdir(parent_dir)
        current_dir = os.getcwd()
        file_paths = f"images/{file.filename}"
        with open(file_paths, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_path =os.path.join(current_dir, f"images/{file.filename}")
        img = cv2.imread(file_path)
        img = cv2.resize(img, (600, 650))
        # cv2.imshow("Image", img)
        text = pytesseract.image_to_string(img)
        words =text.strip().split()
        word_found = False
        for word in words:
            if word.lower() == textt.lower():
                print(f"Found requested word: {word}")
                word_found = True
                break
        
        if not word_found:
            print("Requested word not found")
        response_data = {
            "success": True,
            "message": "OCR performed successfully",
            "detected_words": word,
            "path_parameter": textt,
            "word_found": word_found
        }
        os.remove(file_path)
        return JSONResponse(response_data, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)})
    

    
@router.post("/scanOcr{textt}")
async def scan_file(textt: str):
    try:
         # Initialize the camera
        cap = cv2.VideoCapture(0)  # 0 is usually the default camera
        
        # Capture a single frame
        ret, frame = cap.read()
        
        # Release the camera
        cap.release()
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to capture image")
        img = cv2.imread(frame)
        img = cv2.resize(img, (600, 650))
        # cv2.imshow("Image", img)
        text = pytesseract.image_to_string(img)
        words =text.strip().split()
        word_found = False
        for word in words:
            if word.lower() == textt.lower():
                print(f"Found requested word: {word}")
                word_found = True
                break
        
        if not word_found:
            print("Requested word not found")
        response_data = {
            "success": True,
            "message": "OCR performed successfully on scanned image",
            "detected_words": word,
            "path_parameter": textt,
            "word_found": word_found
        }

        return JSONResponse(response_data, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)})

