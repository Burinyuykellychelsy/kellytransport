import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from data import df

# Variables à réduire
X = df[["distance", "trafic", "heure", "temps"]]

# Normalisation (obligatoire avant PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA : réduire 4 variables en 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Résultats
print("=== RÉDUCTION DE DIMENSION (PCA) ===")
print(f"Variance expliquée par composante 1 : {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"Variance expliquée par composante 2 : {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"Variance totale expliquée : {sum(pca.explained_variance_ratio_)*100:.2f}%")

# Graphique
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], color="purple", alpha=0.5)
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("PCA : Réduction de 4 variables en 2 dimensions")
plt.tight_layout()
plt.savefig("reduction_pca.png")
print("\nGraphique sauvegardé : reduction_pca.png")
