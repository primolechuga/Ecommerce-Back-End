from fastapi import APIRouter, Request, status, UploadFile, File
from pydantic import BaseModel
from app.services.model_service import ModelService

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

        self.router.add_api_route(
            "/get-text-generator", 
            self.get_text_generator, 
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
    

    async def get_image_classifier(self, image: UploadFile = File(...)):
        # Imprime el nombre de la imagen
        print("Nombre de la imagen:", image.filename)
        # Lee el contenido de la imagen (en bytes)
        contents = await image.read()
        # Para evitar imprimir demasiados bytes, mostramos solo los primeros 100
        print("Contenido de la imagen (primeros 100 bytes):", contents[:100])
        
        return {"result": "Imagen impresa"}
    

    async def get_text_generator(self, request: Request):
        data = await request.json()
        text = data.get("inputText")
        print("Texto recibido:", text)
        return {"result": f"Texto recibido: {text}"}

