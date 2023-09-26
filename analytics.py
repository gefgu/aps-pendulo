import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
sns.set_theme()

df = pd.read_csv("positions.csv")

df = df[100:]

# Entre -1 e 1
df["x"] -= (np.max(df["x"]) + np.min(df["x"])) / 2
df["x"] *= 2 /(np.max(df["x"]) - np.min(df["x"]))

# Converte para metros
df["x"] *= (33/2) / 100

def f(t, a, b, w, phi):
  # x = A e^{-bt} cos(wt - phi)
  return a * np.exp((-b*t)) * np.cos(w * t - phi)

popt, _ = curve_fit(f, df["t"], df["x"])
df["fit"] = f(df["t"], *popt)

sns.scatterplot(df, x="t", y="x", label="Pontos Amostrais")
sns.lineplot(df, x="t", y="fit", label=f"Fit. a={popt[0].round(2)} b={popt[1].round(2)} w={popt[2].round(2)} phi={popt[3].round(2)}", color="red")
plt.xlabel("t (s)")
plt.ylabel("x (m)")
plt.title("x(t)")
plt.legend(loc="best")
plt.show()

with open("resultados.txt", "w+") as file:
  # T = 2 * pi / w
  period = (2 * np.pi)/(popt[2])
  # Q = 2 * pi / (1 - e^{-2 * b * T})
  q = 2 * np.pi / (1 - np.exp(-2 * popt[1] * period))
  file.write(f"Fator de Qualidade: {q}")

