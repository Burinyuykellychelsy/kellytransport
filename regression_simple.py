import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from data import df

# Variables
X = df[["distance"]]  # variable explicative
y = df["temps"]       # variable cible

# Modèle
modele = LinearRegression()
modele.fit(X, y)

# Résultats
print("=== RÉGRESSION LINÉAIRE SIMPLE ===")
print(f"Coefficient (pente)    : {modele.coef_[0]:.4f}")
print(f"Intercept              : {modele.intercept_:.4f}")
print(f"Score R²               : {modele.score(X, y):.4f}")
print(f"\nFormule : Temps = {modele.coef_[0]:.2f} x Distance + {modele.intercept_:.2f}")

# Graphique
plt.figure(figsize=(8, 5))
plt.scatter(df["distance"], y, color="blue", alpha=0.5, label="Données réelles")
plt.plot(df["distance"], modele.predict(X), color="red", label="Droite de régression")
plt.xlabel("Distance (km)")
plt.ylabel("Temps (minutes)")
plt.title("Régression Linéaire Simple : Temps = f(Distance)")
plt.legend()
plt.tight_layout()
plt.savefig("regression_simple.png")
print("\nGraphique sauvegardé : regression_simple.png")
