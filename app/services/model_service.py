from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

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
        Predice la venta semanal para el rango de fechas desde self.last_date hasta la fecha objetivo.
        Si se requiere predecir más de una semana, se devuelve un gráfico en formato PNG codificado en base64
        con la demanda predicha para cada semana del rango. Si se predice sólo una semana, se devuelve la predicción
        final en la escala original.

        Parámetros:
        - fecha: str o datetime, la fecha objetivo para la predicción.
        - temperatura: float, valor de la temperatura para la semana objetivo.
        - es_feriado: bool, indicador de si la semana objetivo es feriado.
        
        Retorna:
        - Si se predicen más de una semana: gráfico en base64 (PNG) con la serie de predicciones.
        - Si se predice una sola semana: Predicción de ventas (float) en la escala original.
        """
        # Convertir la fecha a datetime
        fecha_obj = pd.to_datetime(fecha)
        
        # Calcular la cantidad de semanas a predecir entre la última fecha conocida y la fecha objetivo.
        delta_dias = (fecha_obj - self.last_date).days
        n_weeks = delta_dias // 7
        if n_weeks < 1:
            n_weeks = 1  # Al menos se predice la siguiente semana.
        
        # Inicializar la ventana actual a partir de la última ventana guardada.
        current_window = self.last_window.copy()
        
        # Listas para almacenar predicciones y fechas (se usan solo si se predicen varias semanas)
        predicciones_scaled = []
        fechas_pred = []
        
        # Iterar para predecir semana a semana hasta la fecha objetivo.
        for i in range(n_weeks):
            if i == n_weeks - 1:
                # Para la semana objetivo, se usan los valores proporcionados
                temp_scaled = self.scaler_temp.transform([[temperatura]])[0, 0]
                is_holiday_val = 1.0 if es_feriado else 0.0
            else:
                # Para las semanas intermedias se mantienen los valores de la última fila
                temp_scaled = current_window[-1, 1]
                is_holiday_val = current_window[-1, 2]
            
            # Actualizar la última fila de la ventana
            current_window[-1, 1] = temp_scaled
            current_window[-1, 2] = is_holiday_val
            
            # Preparar la secuencia de entrada para el modelo (agregar dimensión de batch)
            input_seq = np.expand_dims(current_window, axis=0)
            
            # Realizar la predicción (valor escalado de Weekly_Sales)
            pred_scaled = self.model.predict(input_seq, verbose=0)[0, 0]
            
            if n_weeks > 1:
                predicciones_scaled.append(pred_scaled)
                fecha_pred = self.last_date + pd.Timedelta(days=(i + 1) * 7)
                fechas_pred.append(fecha_pred)
            
            # Actualizar la ventana: eliminar la fila más antigua y agregar la nueva fila
            new_row = np.array([pred_scaled, temp_scaled, is_holiday_val]).reshape(1, -1)
            current_window = np.append(current_window[1:], new_row, axis=0)
        
        # Invertir el escalado para obtener la predicción final en la escala original
        pred_ventas = self.scaler_sales.inverse_transform([[pred_scaled]])[0, 0]
        
        # Si se predice sólo una semana, devolver la predicción numérica
        if n_weeks == 1:
            return pred_ventas
        else:
            # Convertir las predicciones a la escala original
            predicciones_original = self.scaler_sales.inverse_transform(
                np.array(predicciones_scaled).reshape(-1, 1)
            ).flatten()
            
            # Generar el gráfico
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(fechas_pred, predicciones_original, marker='o', linestyle='-')
            ax.set_title("Predicción de Ventas Semanales")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Ventas")
            ax.grid(True)
            fig.autofmt_xdate()
            
            # Guardar el gráfico en un buffer, codificar en base64 y devolver la cadena resultante
            from io import BytesIO
            import base64
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            encoded_img = base64.b64encode(buf.read()).decode()
            return encoded_img

    
    
    def get_image_classifier(self, image):
        class_names = ['jeans', 'sofa', 'tshirt', 'tv']
        predicciones = self.model_classfier.predict(image)
        clase_predicha = np.argmax(predicciones, axis=1)
        probabilidades = predicciones[0]

        # Obtener el nombre de la clase predicha usando class_names
        clase_predicha = class_names[clase_predicha[0]]

        return clase_predicha