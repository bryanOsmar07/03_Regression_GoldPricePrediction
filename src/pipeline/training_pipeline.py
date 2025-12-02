# src/pipeline/training_pipeline.py

import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainingPipeline:
    """
    Orquesta el pipeline completo de entrenamiento:
    1. Ingesta de datos (train/test split).
    2. Transformación y feature engineering.
    3. Entrenamiento y evaluación del modelo.
    """

    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self) -> dict:
        """
        Ejecuta todas las etapas del pipeline de entrenamiento.

        Returns
        -------
        dict
            Diccionario con:
            - rutas de artifacts (train, test, preprocessor, features, modelo)
            - métricas del modelo (MAE, RMSE, R², etc.)
            - mejores hiperparámetros encontrados.
        """
        try:
            logging.info("===== INICIO PIPELINE DE ENTRENAMIENTO =====")

            # 1. Data Ingestion
            logging.info(">>> Etapa: Data Ingestion")
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()

            # 2. Data Transformation
            logging.info(">>> Etapa: Data Transformation")
            (
                train_arr,
                test_arr,
                preprocessor_path,
                features_path,
            ) = self.data_transformation.initiate_data_transformation(
                train_path=train_path,
                test_path=test_path,
            )

            # 3. Model Training
            logging.info(">>> Etapa: Model Trainer")
            results = self.model_trainer.initiate_model_trainer(
                train_array=train_arr,
                test_array=test_arr,
            )

            # 4. Agregar rutas de artifacts al resultado
            results.update(
                {
                    "train_path": train_path,
                    "test_path": test_path,
                    "preprocessor_path": preprocessor_path,
                    "features_path": features_path,
                }
            )

            logging.info("===== PIPELINE DE ENTRENAMIENTO FINALIZADO =====")
            return results

        except Exception as e:
            logging.error("Error en TrainingPipeline", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    output = pipeline.run_pipeline()
    print(output)
