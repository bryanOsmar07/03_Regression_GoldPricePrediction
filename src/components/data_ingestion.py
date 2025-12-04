# src/components/data_ingestion.py

import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    """
    Configuración de rutas para la ingesta de datos.
    Ajusta raw_data_path si guardas el CSV en otra ubicación.
    """
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    """
    Se encarga de:
    1. Leer el dataset crudo.
    2. Hacer el split train/test.
    3. Guardar los archivos resultantes.
    """

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Ejecuta el flujo de ingesta de datos.

        Returns
        -------
        tuple[str, str]
            Rutas a train_data_path y test_data_path.
        """
        logging.info("Iniciando proceso de Data Ingestion")

        try:
            # 1. Leer dataset crudo
            csv_path = os.path.join("data", "raw", "gld_price_data.csv")
            logging.info(f"Reading dataset from: {csv_path}")
            df = pd.read_csv(csv_path)
            logging.info(f"Dataset leído correctamente. Shape: {df.shape}")

            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data guardado en artifacts/data.csv")

            # 2. Split train / test
            logging.info("Realizando train-test split (test_size=0.2, random_state=42)")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train/Test split completado")

            logging.info(f"Train shape: {train_set.shape}")
            logging.info(f"Test shape: {test_set.shape}")

            # 3. Guardar resultados
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Data Ingestion finalizado correctamente")
            logging.info(f"Train guardado en: {self.ingestion_config.train_data_path}")
            logging.info(f"Test guardado en: {self.ingestion_config.test_data_path}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.error("Error durante la ingesta de datos", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    # Probar el data_ingestion
    # obj.initiate_data_ingestion()

    # Probar el data_transformation
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    # data_transformation.initiate_data_transformation(train_data,test_data)

    # Probar el model_trainer
    train_arr, test_arr, processor_path, _ = (
        data_transformation.initiate_data_transformation(
            train_data,
            test_data,
        )
    )
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
