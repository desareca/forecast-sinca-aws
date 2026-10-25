"""Módulo de análisis de modelos (Fase 3.5): carga del estado MLflow y figuras.

Capas:
- `load`: lee el estado de MLflow (`mlflow.sqlite` + `mlruns/`) desde S3 o local.
- `plots`: figuras Plotly puras (independientes de Streamlit).
- `streamlit_app`: frontend delgado (widgets + caché).
"""
