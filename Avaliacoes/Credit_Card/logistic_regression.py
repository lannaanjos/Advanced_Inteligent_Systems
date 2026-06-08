# Utilitários
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


from sklearn.linear_model import LogisticRegression

lr_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 500]
}
