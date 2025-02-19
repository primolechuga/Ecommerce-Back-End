from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

class ModelService:
    def __init__(self):
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model('app/services/trained_model.h5')
        
        # Cargar los escaladores
        self.scaler_sales = joblib.load('app/services/scaler_sales.pkl')
        self.scaler_temp  = joblib.load('app/services/scaler_temp.pkl')
        self.model_classfier = load_model('app/services/modelo_clasificacion.h5')
        
        # Cargar la última ventana de datos (forma: [window_size, 3])
        self.last_window = np.load('app/services/last_window.npy')
        
        # Definir la última fecha conocida de la serie de entrenamiento.
        self.last_date = pd.to_datetime("2012-10-26")
    
    def get_time_series(self, fecha, es_feriado, temperatura):
        """
        Predice la venta semanal para la fecha objetivo realizando predicciones iterativas semana a semana
        a partir de la última ventana conocida. Cada predicción se utiliza para alimentar la ventana para
        la siguiente iteración; al final, se devuelve la predicción final en la escala original.
        
        Parámetros:
          - fecha: str o datetime, la fecha objetivo para la predicción.
          - temperatura: float, valor de la temperatura para la semana objetivo.
          - es_feriado: bool, indicador de si la semana objetivo es feriado.
          
        Retorna:
          - Predicción de ventas (float) en la escala original, correspondiente a la semana de la fecha dada.
        """
        # Convertir la fecha a datetime
        fecha_obj = pd.to_datetime(fecha)
        
        # Calcular la cantidad de semanas a predecir entre la última fecha conocida y la fecha objetivo.
        # Se asume que la fecha objetivo es posterior a la última fecha conocida.
        delta_dias = (fecha_obj - self.last_date).days
        n_weeks = delta_dias // 7
        if n_weeks < 1:
            n_weeks = 1  # Al menos se predice la siguiente semana.
        
        # Inicializar la ventana actual a partir de la última ventana guardada.
        current_window = self.last_window.copy()
        pred_scaled = None
        
        # Iterar para predecir semana a semana hasta la fecha objetivo.
        for i in range(n_weeks):
            # Para la semana final (la correspondiente a la fecha objetivo), usar los valores proporcionados.
            if i == n_weeks - 1:
                temp_scaled = self.scaler_temp.transform([[temperatura]])[0, 0]
                is_holiday_val = 1.0 if es_feriado else 0.0
            else:
                # Para las semanas intermedias, se mantienen los valores de la última fila de la ventana.
                temp_scaled = current_window[-1, 1]
                is_holiday_val = current_window[-1, 2]
            
            # Actualizar la última fila de la ventana con la temperatura e indicador de feriado correspondientes.
            current_window[-1, 1] = temp_scaled
            current_window[-1, 2] = is_holiday_val
            
            # Preparar la secuencia de entrada para el modelo (agregar dimensión de batch).
            input_seq = np.expand_dims(current_window, axis=0)
            
            # Realizar la predicción para la siguiente semana (valor escalado de Weekly_Sales).
            pred_scaled = self.model.predict(input_seq, verbose=0)[0, 0]
            
            # Actualizar la ventana: eliminar la fila más antigua y agregar la nueva fila con la predicción
            # y los valores de temperatura e indicador de feriado correspondientes.
            new_row = np.array([pred_scaled, temp_scaled, is_holiday_val]).reshape(1, -1)
            current_window = np.append(current_window[1:], new_row, axis=0)
        
        # Invertir el escalado para obtener la predicción final en la escala original de ventas.
        pred_ventas = self.scaler_sales.inverse_transform([[pred_scaled]])[0, 0]
        
        return pred_ventas
    
    def get_image_classifier(self, image):
        class_names = ['jeans', 'sofa', 'tshirt', 'tv']
        predicciones = self.model_classfier.predict(image)
        clase_predicha = np.argmax(predicciones, axis=1)
        probabilidades = predicciones[0]

        # Obtener el nombre de la clase predicha usando class_names
        clase_predicha = class_names[clase_predicha[0]]

        return clase_predicha