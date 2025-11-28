# src/utils.py

import os
import pickle
import sys

from src.exception import CustomException


def save_object(file_path: str, obj):
    """
    Guarda cualquier objeto serializable en el path indicado usando pickle.
    - Crea los directorios si no existen
    - Maneja excepciones propias del pipeline
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Carga un objeto pickle desde el path especificado.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)