# app.py

import os
from datetime import datetime

import pandas as pd
import streamlit as st

from src.pipeline.predict_pipeline import PredictPipeline

st.set_page_config(
    page_title="Gold Price Prediction",
    page_icon="🪙",
    layout="centered",
)

st.title("🪙 Gold Price Prediction - Versión 2.0")
st.markdown(
    """
Esta versión solo te pide:

**Fecha + SPX + USO + SLV + EUR/USD**

y la aplicación se encarga de:

- Cargar el histórico de mercado.
- Construir automáticamente las features:
  - `year`, `month`, `week`, `dayofyear`
  - `ret_SPX`, `ret_USO`, `ret_SLV`, `ret_EURUSD`
  - `vol_SPX_7`
- Enviar esas features al modelo entrenado para estimar el precio de **GLD**.
"""
)

st.markdown("---")


@st.cache_data
def load_historical_data() -> pd.DataFrame:
    """
    Carga el dataset histórico que se usó para entrenar el modelo.
    Intenta primero con artifacts/data.csv (creado en DataIngestion),
    y si no existe, usa data/raw/gld_price_data.csv.
    """
    candidates = [
        os.path.join("artifacts", "data.csv"),
        os.path.join("data", "raw", "gld_price_data.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df

    raise FileNotFoundError(
        "No se encontró el dataset histórico. "
        "Revisa si existe artifacts/data.csv o data/raw/gld_price_data.csv."
    )


def build_features_from_input(
    date_str: str,
    spx: float,
    uso: float,
    slv: float,
    eur_usd: float,
) -> pd.DataFrame:
    """
    A partir de la fecha y los valores ingresados,
    construye un DataFrame de una fila con todas las features
    que el modelo espera, usando el histórico para calcular
    retornos y volatilidad.
    """
    # 1. Cargar histórico
    df_hist = load_historical_data().copy()

    # Asegurar formato de columnas
    if "EUR/USD" not in df_hist.columns:
        raise ValueError("La columna 'EUR/USD' no está en el dataset histórico.")

    # 2. Convertir Date a datetime y ordenar por fecha
    df_hist["Date"] = pd.to_datetime(df_hist["Date"], format="%m/%d/%Y")
    df_hist = df_hist.sort_values("Date").reset_index(drop=True)

    # 3. Crear una nueva fila con la info del usuario (sin features aún)
    new_date = datetime.strptime(date_str, "%Y-%m-%d")  # streamlit
    new_row = {
        "Date": new_date,
        "SPX": spx,
        "USO": uso,
        "SLV": slv,
        "EUR/USD": eur_usd,
        # GLD se desconoce en producción, no se usa aquí
    }

    # 4. Concatenar histórico + nueva fila
    df_all = pd.concat([df_hist, pd.DataFrame([new_row])], ignore_index=True)

    # 5. Reordenar por fecha (por si la fecha ingresada no es la última)
    df_all = df_all.sort_values("Date").reset_index(drop=True)

    # 6. Feature engineering (igual que en data_transformation, sin derivar de GLD)
    df_all["year"] = df_all["Date"].dt.year
    df_all["month"] = df_all["Date"].dt.month
    df_all["day"] = df_all["Date"].dt.day
    df_all["dayofyear"] = df_all["Date"].dt.dayofyear
    df_all["week"] = df_all["Date"].dt.isocalendar().week.astype(int)

    # Retornos porcentuales
    df_all["ret_SPX"] = df_all["SPX"].pct_change()
    df_all["ret_USO"] = df_all["USO"].pct_change()
    df_all["ret_SLV"] = df_all["SLV"].pct_change()
    df_all["ret_EURUSD"] = df_all["EUR/USD"].pct_change()

    # Volatilidad rolling 7 días para SPX
    df_all["vol_SPX_7"] = df_all["SPX"].pct_change().rolling(7).std()

    # 7. Tomar la fila correspondiente a la fecha ingresada (ya con features)
    df_all = df_all.reset_index(drop=True)
    mask = df_all["Date"] == new_date
    df_input = df_all.loc[mask].copy()

    if df_input.empty:
        raise ValueError("No se pudo encontrar la fila correspondiente.")

    # 8. Seleccionar solo las columnas de features
    feature_cols = [
        "SLV", "EUR/USD", "SPX", "USO",
        "year", "month", "week", "dayofyear",
        "ret_SPX", "ret_USO", "ret_SLV", "ret_EURUSD",
        "vol_SPX_7",
    ]
    df_input = df_input[feature_cols]

    return df_input


# ---------- UI ----------

st.header("📥 Ingreso de datos básicos")

with st.form("prediction_form_v2"):
    col1, col2 = st.columns(2)

    with col1:
        date_input = st.date_input(
            "Fecha (Date)",
            value=datetime(2013, 3, 14),
            format="YYYY-MM-DD",
        )
        SPX = st.number_input(
            "SPX (S&P 500)",
            min_value=0.0,
            value=1500.0,
            step=1.0,
        )
        USO = st.number_input(
            "USO (precio petróleo, ETF USO)",
            min_value=0.0,
            value=30.0,
            step=0.1,
        )

    with col2:
        SLV = st.number_input(
            "SLV (precio plata, ETF SLV)",
            min_value=0.0,
            value=17.0,
            step=0.1,
        )
        EUR_USD = st.number_input(
            "EUR/USD (tipo de cambio)",
            min_value=0.5,
            value=1.20,
            step=0.001,
            format="%.3f",
        )

    submitted = st.form_submit_button("🔮 Predecir precio del oro (GLD)")

if submitted:
    try:
        # 1. Construir DataFrame de features automáticamente
        date_str = date_input.strftime("%Y-%m-%d")
        df_features = build_features_from_input(
            date_str=date_str,
            spx=SPX,
            uso=USO,
            slv=SLV,
            eur_usd=EUR_USD,
        )

        st.markdown("### 🔎 Features generadas automáticamente")
        st.dataframe(df_features)

        # 2. Ejecutar pipeline de predicción
        pipeline = PredictPipeline()
        pred = pipeline.predict(df_features)

        st.markdown("---")
        st.subheader("📊 Resultado de la predicción")
        st.success(f"Precio estimado de GLD: **{pred[0]:.2f}**")

    except Exception as e:
        st.error("Ocurrió un error durante la predicción.")
        st.exception(e)

st.markdown("---")
st.caption(
    "Proyecto: 03_Regression_GoldPricePrediction · "
    "Versión 2.0 (auto-feature engineering)"
)
