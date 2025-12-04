# tests/test_data_ingestion.py

import os

import pandas as pd

from src.components.data_ingestion import DataIngestion


def test_data_ingestion_creates_train_and_test_files():
    """Verifica que DataIngestion genera train.csv y test.csv correctamente."""
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Archivos existen
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)

    # Leemos los CSV
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # No están vacíos
    assert not train_df.empty
    assert not test_df.empty

    # Tienen las mismas columnas
    assert set(train_df.columns) == set(test_df.columns)

    # Debe existir la columna objetivo GLD
    assert "GLD" in train_df.columns
