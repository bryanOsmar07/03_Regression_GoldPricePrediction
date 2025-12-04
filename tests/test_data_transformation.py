# tests/test_data_transformation.py

import os
import numpy as np

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


def test_data_transformation_shapes_and_artifacts():
    """Verifica que la transformación produce arrays y artifacts válidos."""
    # 1. Ingesta
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # 2. Transformación
    transformer = DataTransformation()
    (
        train_arr,
        test_arr,
        preprocessor_path,
        features_path,
    ) = transformer.initiate_data_transformation(
        train_path=train_path,
        test_path=test_path,
    )

    # Arrays 2D
    assert isinstance(train_arr, np.ndarray)
    assert isinstance(test_arr, np.ndarray)
    assert train_arr.ndim == 2
    assert test_arr.ndim == 2

    # Misma cantidad de columnas
    assert train_arr.shape[1] == test_arr.shape[1]

    # La última columna (target) debe tener varianza > 0
    assert np.std(train_arr[:, -1]) > 0

    # Artifacts existen
    assert os.path.exists(preprocessor_path)
    assert os.path.exists(features_path)
