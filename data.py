import pandas as pd
import numpy as np

# Données simulées de trajets
np.random.seed(42)

n = 100  # 100 trajets

data = {
    "distance": np.random.uniform(5, 200, n),
    "heure": np.random.randint(0, 24, n),
    "trafic": np.random.uniform(1, 10, n),
}

# Temps de trajet calculé (en minutes)
data["temps"] = (
    data["distance"] * 0.5
    + data["trafic"] * 3
    + data["heure"] * 0.8
    + np.random.normal(0, 5, n)
)

df = pd.DataFrame(data)

print(df.head())
