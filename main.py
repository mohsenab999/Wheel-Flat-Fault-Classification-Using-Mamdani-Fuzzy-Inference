import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from mamdani_inference_engine import mamdani_inference

wheel_flat_classifier = mamdani_inference()

data_dir = Path(__file__).resolve().parent
df = pd.read_csv(data_dir / "WheelFlat_Dataset2_Test (2).csv")
df.columns = df.columns.str.strip()

y = df["kurtosis"].to_numpy()
x = df["rms_g"].to_numpy()
z = df["label"].to_numpy()

for i in range(30):
    input_data = [x[i], y[i], z[i]]
    wheel_flat_classifier.classifier(input_data)