# tests/test_predict_pipeline.py

import os

import numpy as np
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils import load_object


def test_predict_pipeline_returns_single_prediction():
    """
    Verifica que el PredictPipeline genere una predicción válida
    a partir de un DataFrame de features.
    """
    # 1. Aseguramos que el modelo esté entrenado y artifacts generados
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()

    # 2. Tomamos un ejemplo del histórico (train.csv)
    ingestion = DataIngestion()
    train_path, _ = ingestion.initiate_data_ingestion()
    train_df = pd.read_csv(train_path)

    # 3. Aplicar el mismo feature engineering que en DataTransformation
    transformer = DataTransformation()
    train_fe = transformer._feature_engineering(train_df)

    # 4. Cargar la lista de features esperadas
    features_path = os.path.join("artifacts", "features.pkl")
    feature_cols = load_object(features_path)

    # 5. Construir un DataFrame con una sola fila de features
    sample_df = train_fe[feature_cols].iloc[[0]]  # DataFrame de 1 fila

    # 6. Pipeline de predicción
    predict_pipeline = PredictPipeline()
    preds = predict_pipeline.predict(sample_df)

    # 7. Verificaciones
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (1,)
    assert isinstance(preds[0], (float, int, np.floating))
