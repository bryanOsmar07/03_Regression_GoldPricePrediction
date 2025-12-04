# tests/test_model_trainer.py

import os
import math

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def test_model_trainer_runs_and_saves_model():
    """Verifica que ModelTrainer entrena y guarda un modelo con métricas válidas."""
    # 1. Ingesta
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # 2. Transformación
    transformer = DataTransformation()
    train_arr, test_arr, preprocessor_path, features_path = (
        transformer.initiate_data_transformation(
            train_path=train_path,
            test_path=test_path,
        )
    )

    # 3. Entrenamiento
    trainer = ModelTrainer()
    results = trainer.initiate_model_trainer(
        train_array=train_arr,
        test_array=test_arr,
    )

    # Claves esperadas
    for key in [
        "model_path",
        "best_params",
        "cv_best_score_neg_mse",
        "test_mae",
        "test_rmse",
        "test_r2",
    ]:
        assert key in results

    # Modelo guardado
    assert os.path.exists(results["model_path"])

    # R2 en rango [-1, 1] y no NaN
    r2 = results["test_r2"]
    assert -1.0 <= r2 <= 1.0
    assert not math.isnan(r2)
