from fastapi import APIRouter, Request, status, UploadFile, File
from pydantic import BaseModel
from app.services.model_service import ModelService
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import json
from fastapi.responses import JSONResponse
class ModelController:

    def __init__(self):
        self.router = APIRouter(prefix="/api")
        self.model_service = ModelService()
        
        self.router.add_api_route(
            "/get-time-series", 
            self.get_time_series, 
            methods=["POST"],
            status_code=status.HTTP_200_OK
        )

        self.router.add_api_route(
            "/get-image-classifier", 
            self.get_image_classifier, 
            methods=["POST"],
            status_code=status.HTTP_200_OK
        )




    async def get_time_series(self, request: Request):
        # Se obtiene el JSON del request
        data = await request.json()
        
        # Desempaquetamos los valores del JSON.
        date = data.get("date")
        holiday = data.get("isHoliday")
        temperature = data.get("temperature")
        
        print("Fecha:", date, "Holiday:", holiday, "Temperature:", temperature)
        
        # Llamamos al método del servicio que realiza las predicciones iterativas.
        result = self.model_service.get_time_series(date, holiday, temperature)
        print("Resultado predicción:", result)
        
        return {"result": f"$ {int(result)}"}
    

    async def get_image_classifier(self, image_file: UploadFile = File(...)):
        # Leer la imagen en bytes y abrirla con PIL
        contents = await image_file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")  # Convertir a RGB por compatibilidad

        # Convertir la imagen a array de NumPy
        img = img.resize((224, 224))  # Ajustar tamaño según el modelo
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión batch
        img_array /= 255.0  # Normalizar

        prediction = self.model_service.get_image_classifier(img_array)

        print("Prediction result:", prediction)

        return prediction


        
