# src/components/data_transformation.py

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """
    Rutas de salida de la etapa de transformación.
    - preprocessor_obj_file_path: objeto de preprocesamiento de sklearn
    - features_obj_file_path: lista de columnas usadas como features
    """
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    features_obj_file_path: str = os.path.join("artifacts", "features.pkl")


class DataTransformation:
    """
    Se encarga de transformar los datos crudos en matrices listas
    para el modelado (features + target).
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        # Variables finales definidas en el EDA (features del modelo)
        self.feature_cols = [
            "SLV", "EUR/USD", "SPX", "USO",
            "year", "month", "week", "dayofyear",
            "ret_SPX", "ret_USO", "ret_SLV", "ret_EURUSD", "ret_GLD",
            "vol_SPX_7", "vol_GLD_7"
        ]
        self.target_col = "GLD"

    # ---------- 1) Objeto de preprocesamiento ----------
    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Crea y devuelve el objeto de preprocesamiento de sklearn.
        En este proyecto solo usamos variables numéricas, así que
        aplicamos imputación (median) sobre self.feature_cols.
        """

        logging.info("Creando objeto de transformación de datos")

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                # Si más adelante quisieras usar modelos lineales,
                # aquí podrías añadir un StandardScaler.
                # ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.feature_cols),
            ],
            remainder="drop",
        )

        return preprocessor

    # ---------- 2) Feature engineering ----------
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica las transformaciones de fecha y creación de nuevas variables,
        tal como se definió en el notebook.
        """

        logging.info("Iniciando feature engineering sobre dataframe")

        # Aseguramos tipo datetime
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

        # Variables de calendario
        df["year"] = df["Date"].dt.year
        df["month"] = df["Date"].dt.month
        df["day"] = df["Date"].dt.day
        df["dayofyear"] = df["Date"].dt.dayofyear
        df["week"] = df["Date"].dt.isocalendar().week.astype(int)

        # Retornos porcentuales
        df["ret_SPX"] = df["SPX"].pct_change()
        df["ret_USO"] = df["USO"].pct_change()
        df["ret_SLV"] = df["SLV"].pct_change()
        df["ret_EURUSD"] = df["EUR/USD"].pct_change()
        df["ret_GLD"] = df["GLD"].pct_change()

        # Volatilidades rolling 7 días
        df["vol_SPX_7"] = df["SPX"].pct_change().rolling(7).std()
        df["vol_GLD_7"] = df["GLD"].pct_change().rolling(7).std()

        # Eliminamos filas con NaN generados por pct_change/rolling
        df = df.dropna().reset_index(drop=True)

        logging.info(f"Feature engineering completado. Shape: {df.shape}")
        return df

    # ---------- 3) Orquestador ----------
    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Orquesta la transformación:
        1. Lee train/test desde artifacts.
        2. Aplica feature engineering.
        3. Separa X/y.
        4. Ajusta y aplica el preprocessor.
        5. Guarda preprocessor y lista de features.

        Parameters
        ----------
        train_path : str
            Ruta al CSV de entrenamiento (artifacts/train.csv).
        test_path : str
            Ruta al CSV de prueba (artifacts/test.csv).

        Returns
        -------
        tuple
            X_train_array, y_train_array, X_test_array, y_test_array,
            ruta_preprocessor, ruta_features
        """
        try:
            # 1. Lectura de train y test
            logging.info("Leyendo datasets de train y test")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Train shape original: {train_df.shape}")
            logging.info(f"Test shape original: {test_df.shape}")

            logging.info("Aplicando feature engineering a train y test")
            train_df = self._feature_engineering(train_df)
            test_df = self._feature_engineering(test_df)

            logging.info(f"Train transformado shape: {train_df.shape}")
            logging.info(f"Test transformado shape: {test_df.shape}")

            # 2. Separar X / y
            X_train = train_df[self.feature_cols]
            y_train = train_df[self.target_col]

            X_test = test_df[self.feature_cols]
            y_test = test_df[self.target_col]

            logging.info("Creando objeto preprocessor")
            preprocessor = self.get_data_transformer_object()

            logging.info("Ajustando preprocessor con datos de train")
            X_train_array = preprocessor.fit_transform(X_train)
            X_test_array = preprocessor.transform(X_test)

            # Guardar el preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )
            logging.info(
                f"Preprocessor guardado en: "
                f"{self.data_transformation_config.preprocessor_obj_file_path}"
            )

            # Guardar la lista de features usadas (útil para inferencia/explicación)
            save_object(
                file_path=self.data_transformation_config.features_obj_file_path,
                obj=self.feature_cols,
            )
            logging.info(
                f"Lista de features guardada en: "
                f"{self.data_transformation_config.features_obj_file_path}"
            )

            # Unir X e y en un solo array para pasarlo al trainer
            train_arr = np.c_[X_train_array, y_train.to_numpy()]
            test_arr = np.c_[X_test_array, y_test.to_numpy()]

            logging.info(f"Shape final train_arr: {train_arr.shape}")
            logging.info(f"Shape final test_arr: {test_arr.shape}")
            logging.info("Data Transformation finalizado correctamente")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.features_obj_file_path,
            )

        except Exception as e:
            logging.error("Error en data transformation", exc_info=True)
            raise CustomException(e, sys)