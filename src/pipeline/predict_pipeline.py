# src/pipeline/predict_pipeline.py

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


@dataclass
class CustomData:
    """
    Representa una nueva observación con las mismas features
    que el modelo espera.

    NOTA:
    - No se usan variables derivadas del target (GLD), solo
      variables exógenas y de calendario + retornos/volatilidad
      de los índices SPX, USO, SLV y EUR/USD.
    """

    SLV: float
    EUR_USD: float  # columna 'EUR/USD'
    SPX: float
    USO: float
    year: int
    month: int
    week: int
    dayofyear: int
    ret_SPX: float
    ret_USO: float
    ret_SLV: float
    ret_EURUSD: float
    vol_SPX_7: float

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Convierte los atributos en un DataFrame con los nombres
        de columnas esperados por el preprocessor/modelo.
        """
        try:
            data_dict = {
                "SLV": [self.SLV],
                "EUR/USD": [self.EUR_USD],
                "SPX": [self.SPX],
                "USO": [self.USO],
                "year": [self.year],
                "month": [self.month],
                "week": [self.week],
                "dayofyear": [self.dayofyear],
                "ret_SPX": [self.ret_SPX],
                "ret_USO": [self.ret_USO],
                "ret_SLV": [self.ret_SLV],
                "ret_EURUSD": [self.ret_EURUSD],
                "vol_SPX_7": [self.vol_SPX_7],
            }

            df = pd.DataFrame(data_dict)
            return df

        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    """
    Pipeline de predicción:
    1. Carga el preprocessor y el modelo desde artifacts.
    2. Aplica el preprocessor a los datos de entrada.
    3. Devuelve la predicción de GLD.
    """

    def __init__(
        self,
        model_path: str | None = None,
        preprocessor_path: str | None = None,
        features_path: str | None = None,
    ):
        self.model_path = model_path or os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = preprocessor_path or os.path.join(
            "artifacts", "preprocessor.pkl"
        )
        self.features_path = features_path or os.path.join(
            "artifacts", "features.pkl"
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Aplica el preprocessor y el modelo entrenado a un DataFrame
        de features y devuelve las predicciones.

        Parameters
        ----------
        features : pd.DataFrame
            DataFrame con las columnas esperadas por el preprocessor.

        Returns
        -------
        np.ndarray
            Predicciones del modelo (valores de GLD).
        """
        try:
            logging.info("Iniciando predicción en PredictPipeline")

            # 1. Cargar objetos
            logging.info(f"Cargando preprocessor desde: {self.preprocessor_path}")
            preprocessor = load_object(self.preprocessor_path)

            logging.info(f"Cargando modelo desde: {self.model_path}")
            model = load_object(self.model_path)

            # 2. (Opcional pero recomendable) Validar columnas usando features.pkl
            if os.path.exists(self.features_path):
                expected_features = load_object(self.features_path)
                missing = set(expected_features) - set(features.columns)
                if missing:
                    logging.warning(
                        f"El DataFrame de entrada no contiene todas las columnas "
                        f"esperadas. Faltan: {missing}"
                    )
                # Reordenar columnas para que coincidan con el orden esperado
                features = features[expected_features]

            # 3. Transformar y predecir
            logging.info("Aplicando preprocessor a los datos de entrada")
            data_scaled = preprocessor.transform(features)

            logging.info("Generando predicciones")
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            logging.error("Error en PredictPipeline", exc_info=True)
            raise CustomException(e, sys)
