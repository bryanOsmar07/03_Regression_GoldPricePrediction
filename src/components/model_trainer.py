# src/components/model_trainer.py

import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    """
    Rutas de salida de la etapa de entrenamiento del modelo.
    - trained_model_file_path: ruta donde se guardará el mejor modelo.
    """

    trained_model_file_path: str = os.path.join("artifacts", "model_rf_tuned.pkl")


class ModelTrainer:
    """
    Se encarga de:
    1. Recibir los arrays de train y test (con la última columna como target).
    2. Entrenar un modelo de Machine Learning (RandomForestRegressor).
    3. Ajustar hiperparámetros con RandomizedSearchCV.
    4. Evaluar el modelo sobre el set de test.
    5. Guardar el mejor modelo entrenado.
    """

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def _get_base_model(self) -> RandomForestRegressor:
        """
        Devuelve el modelo base a entrenar.
        """
        return RandomForestRegressor(random_state=42, n_jobs=-1)

    def _get_param_distributions(self) -> dict:
        """
        Define el espacio de búsqueda de hiperparámetros para RandomizedSearchCV.
        (Basado en lo que ya probaste en el notebook).
        """
        param_dist = {
            "n_estimators": [200, 300, 400, 500],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.5, 0.7],
        }
        return param_dist

    def initiate_model_trainer(
        self,
        train_array: np.ndarray,
        test_array: np.ndarray,
    ) -> dict:
        """
        Entrena el modelo de Machine Learning usando los datos transformados
        y evalúa su desempeño sobre el conjunto de prueba.

        Parameters
        ----------
        train_array : np.ndarray
            Array de entrenamiento donde la última columna es el target.
        test_array : np.ndarray
            Array de prueba donde la última columna es el target.

        Returns
        -------
        dict
            Diccionario con las métricas del modelo y los mejores hiperparámetros.
        """
        try:
            logging.info("Iniciando entrenamiento del modelo")

            # 1. Separar X e y
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logging.info(f"Shape X_train: {X_train.shape}, y_train: {y_train.shape}")
            logging.info(f"Shape X_test: {X_test.shape}, y_test: {y_test.shape}")

            # 2. Modelo base y espacio de hiperparámetros
            base_model = self._get_base_model()
            param_dist = self._get_param_distributions()

            logging.info("Iniciando RandomizedSearchCV para RandomForestRegressor")

            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_dist,
                n_iter=30,  # puedes subirlo si quieres más fine-tuning
                scoring="neg_mean_squared_error",
                cv=5,
                verbose=1,
                random_state=42,
                n_jobs=-1,
            )

            # 3. Ajuste de hiperparámetros
            random_search.fit(X_train, y_train)

            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            best_score = random_search.best_score_

            logging.info(f"Mejores hiperparámetros encontrados: {best_params}")
            logging.info(f"Mejor score (neg MSE) en CV: {best_score}")

            # 4. Evaluación en test
            y_pred = best_model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            logging.info("Evaluación del modelo en conjunto de test")
            logging.info(f"MAE  : {mae:.4f}")
            logging.info(f"RMSE : {rmse:.4f}")
            logging.info(f"R2   : {r2:.6f}")

            # 5. Guardar el modelo entrenado
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logging.info(
                f"Modelo entrenado guardado en: "
                f"{self.model_trainer_config.trained_model_file_path}"
            )

            # 6. Devolver resumen de métricas e hiperparámetros
            results = {
                "model_path": self.model_trainer_config.trained_model_file_path,
                "best_params": best_params,
                "cv_best_score_neg_mse": best_score,
                "test_mae": mae,
                "test_rmse": rmse,
                "test_r2": r2,
            }

            logging.info("Entrenamiento del modelo finalizado correctamente")
            return results

        except Exception as e:
            logging.error("Error en model_trainer", exc_info=True)
            raise CustomException(e, sys)
