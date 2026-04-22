import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from data import df

# Variables explicatives (3 variables cette fois)
X = df[["distance", "trafic", "heure"]]
y = df["temps"]

# Modèle
modele = LinearRegression()
modele.fit(X, y)

# Résultats
print("=== RÉGRESSION LINÉAIRE MULTIPLE ===")
print(f"Coefficient distance : {modele.coef_[0]:.4f}")
print(f"Coefficient trafic : {modele.coef_[1]:.4f}")
print(f"Coefficient heure : {modele.coef_[2]:.4f}")
print(f"Intercept : {modele.intercept_:.4f}")
print(f"Score R² : {modele.score(X, y):.4f}")
print(f"\nFormule : Temps = {modele.coef_[0]:.2f}*Distance + {modele.coef_[1]:.2f}*Trafic + {modele.coef_[2]:.2f}*Heure + {modele.intercept_:.2f}")

# Graphique : valeurs réelles vs prédites
y_pred = modele.predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(y, y_pred, color="green", alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--", label="Prédiction parfaite")
plt.xlabel("Temps réel (min)")
plt.ylabel("Temps prédit (min)")
plt.title("Régression Linéaire Multiple : Réel vs Prédit")
plt.legend()
plt.tight_layout()
plt.savefig("regression_multiple.png")
print("\nGraphique sauvegardé : regression_multiple.png")
