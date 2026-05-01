import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



df = pd.read_csv("WheelFlat_Dataset1_Train (2).csv")

y = df["kurtosis"].to_numpy()
x = df["rms_g"].to_numpy()
z = df["label"].to_numpy()

x_min = np.min(x)
x_max = np.max(x)

y_min = np.min(y)
y_max = np.max(y)

plt.scatter(x[z == 0], y[z == 0], color="red",   label="Class 0")
plt.scatter(x[z == 1], y[z == 1], color="blue",  label="Class 1")
plt.scatter(x[z == 2], y[z == 2], color="green", label="Class 2")
plt.ylabel("kurtosis")
plt.xlabel("rms_g")
plt.legend()
plt.show()

print(f"the max and min value of  rms_g data  is {x_min}and {x_max} respectively")
print(f"the max and min value of kurtosis data  is {y_min}and {y_max} respectively")
