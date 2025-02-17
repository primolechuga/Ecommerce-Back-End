from fastapi import APIRouter, Request, status
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

        

    async def get_time_series(self, request: Request):
        # Se obtiene el JSON del request
        data = await request.json()
        
        # Desempaquetamos los valores del JSON.
        # Alternativamente, podrías usar TimeSeriesRequest.parse_obj(data)
        date = data.get("date")
        holiday = data.get("isHoliday")
        temperature = data.get("temperature")
        
        print(date, holiday, temperature)
        
        # Llamamos al método del servicio que realiza las predicciones iterativas.
        result = self.model_service.get_time_series(date, holiday, temperature)
        print(result)
        
        # Devolvemos el resultado (se envuelve en un diccionario para FastAPI)
        return {"result": f"$ {int(result)}"}
    

