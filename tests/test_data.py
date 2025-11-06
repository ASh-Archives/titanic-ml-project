import pytest
import pandas as pd
from src.data import preprocess_data

def test_preprocess_data():
    df = pd.read_csv('data/raw/train.csv')
    X, y = preprocess_data(df)
    assert X.shape[0] == df.shape[0]
    assert X.shape[1] == 9  # Ожидаем 9 признаков после предобработки