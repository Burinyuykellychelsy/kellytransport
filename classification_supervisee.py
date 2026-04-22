import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data import df

# Créer les catégories manuellement
def classer_trajet(temps):
    if temps < 30:
        return "Rapide"
    elif temps < 60:
        return "Moyen"
    else:
        return "Lent"

df["categorie"] = df["temps"].apply(classer_trajet)

# Variables
X = df[["distance", "trafic", "heure"]]
y = df["categorie"]

# Diviser en données d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle KNN
modele = KNeighborsClassifier(n_neighbors=3)
modele.fit(X_train, y_train)

# Résultats
y_pred = modele.predict(X_test)

print("=== CLASSIFICATION SUPERVISÉE (KNN) ===")
print(classification_report(y_test, y_pred))

# Graphique
couleurs = {"Rapide": "green", "Moyen": "orange", "Lent": "red"}
plt.figure(figsize=(8, 5))
for cat, couleur in couleurs.items():
    subset = df[df["categorie"] == cat]
    plt.scatter(subset["distance"], subset["temps"], 
                color=couleur, label=cat, alpha=0.6)

plt.xlabel("Distance (km)")
plt.ylabel("Temps (min)")
plt.title("Classification Supervisée : Rapide / Moyen / Lent")
plt.legend()
plt.tight_layout()
plt.savefig("classification_supervisee.png")
print("\nGraphique sauvegardé : classification_supervisee.png")
