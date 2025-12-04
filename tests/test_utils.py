# tests/test_utils.py

import os

import pytest

from src.exception import CustomException
from src.utils import load_object, save_object


def test_save_and_load_object(tmp_path):
    """Verifica que guardar y cargar un objeto funciona correctamente."""
    obj = {"a": 1, "b": [1, 2, 3]}
    file_path = tmp_path / "obj.pkl"

    # Guardar
    save_object(str(file_path), obj)

    assert os.path.exists(file_path)

    # Cargar
    loaded = load_object(str(file_path))

    assert loaded == obj


def test_load_object_non_existent_raises_custom_exception(tmp_path):
    """Verifica que cargar un path inexistente lanza CustomException."""
    fake_path = tmp_path / "no_existe.pkl"

    with pytest.raises(CustomException):
        load_object(str(fake_path))
