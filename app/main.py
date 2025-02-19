# main.py
import os
from fastapi import FastAPI
import uvicorn
from app.controllers.model_controller import ModelController
from app.middlewares.cors_middleware import CORS

class App:
    def __init__(self):
        self.app = FastAPI()
        self.cors_middleware = CORS()
        self._configure_middlewares()
        self._configure_routes()

    def _configure_middlewares(self):
        self.cors_middleware.add(self.app)

    def _configure_routes(self):
        model_controller = ModelController()

        self.app.include_router(model_controller.router)

    def get_app(self):
        return self.app

app_instance = App()
app = app_instance.get_app()

# Ejecución del servidor
if __name__ == "__main__":
    # Obtén el puerto desde la variable de entorno o usa 8000 por defecto
    port = int(os.getenv("PORT", 8000))
    # Ejecuta el servidor con el host 0.0.0.0
    uvicorn.run(app, host="0.0.0.0", port=port)