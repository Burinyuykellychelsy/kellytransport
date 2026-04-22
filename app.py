from flask import Flask, render_template, request, jsonify
import json, os, uuid
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
RECEIPTS_FILE = "receipts.json"

# ============================
# GESTION DES RECUS
# ============================
def load_receipts():
    if os.path.exists(RECEIPTS_FILE):
        with open(RECEIPTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_receipts(receipts):
    with open(RECEIPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(receipts, f, ensure_ascii=False, indent=2)

# ============================
# TARIFS PAR VEHICULE
# ============================
VEHICLE_CONFIG = {
    "moto": {"base": 500, "per_km": 150, "vitesse": 60},
    "voiture": {"base": 1000, "per_km": 300, "vitesse": 80},
    "camion": {"base": 2000, "per_km": 500, "vitesse": 50},
    "bus": {"base": 1500, "per_km": 400, "vitesse": 70},
    "cheval": {"base": 300, "per_km": 100, "vitesse": 20},
    "vip": {"base": 5000, "per_km": 800, "vitesse": 120},
}

def calculer_tarif(distance, vehicule, trafic, moment):
    cfg = VEHICLE_CONFIG[vehicule]
    tarif = cfg["base"] + cfg["per_km"] * distance
    if trafic == "dense":
        tarif *= 1.3
    elif trafic == "moyen":
        tarif *= 1.1
    if moment == "nuit":
        tarif *= 1.2
    vitesse = cfg["vitesse"]
    if trafic == "dense":
        vitesse *= 0.6
    elif trafic == "moyen":
        vitesse *= 0.8
    duree = (distance / vitesse) * 60
    return round(tarif), round(duree)

# ============================
# ROUTES
# ============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/receipts", methods=["GET"])
def get_receipts():
    return jsonify(load_receipts())

@app.route("/api/receipt", methods=["POST"])
def create_receipt():
    data = request.json
    distance = float(data["distance"])
    vehicule = data["vehicule"]
    trafic = data["trafic"]
    moment = data["moment"]
    tarif, duree = calculer_tarif(distance, vehicule, trafic, moment)
    recu = {
        "id": str(uuid.uuid4())[:8],
        "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "nom": data["nom"],
        "prenom": data["prenom"],
        "age": data["age"],
        "contact": data["contact"],
        "depart": data["depart"],
        "destination": data["destination"],
        "distance": distance,
        "vehicule": vehicule,
        "trafic": trafic,
        "moment": moment,
        "duree": duree,
        "tarif": tarif
    }
    receipts = load_receipts()
    receipts.append(recu)
    save_receipts(receipts)
    return jsonify(recu)

@app.route("/api/receipt/<rid>", methods=["DELETE"])
def delete_receipt(rid):
    receipts = load_receipts()
    receipts = [r for r in receipts if r["id"] != rid]
    save_receipts(receipts)
    return jsonify({"status": "ok"})

@app.route("/api/stats", methods=["GET"])
def get_stats():
    receipts = load_receipts()
    if len(receipts) < 3:
        return jsonify({"error": "Pas assez de donnees (minimum 3 recus)"})

    distances = np.array([r["distance"] for r in receipts]).reshape(-1, 1)
    tarifs = np.array([r["tarif"] for r in receipts])

    # -- Regression lineaire simple --
    lr = LinearRegression()
    lr.fit(distances, tarifs)
    r2_simple = round(lr.score(distances, tarifs), 4)
    coef = round(float(lr.coef_[0]), 2)
    intercept = round(float(lr.intercept_), 2)

    # -- Regression lineaire multiple --
    trafic_map = {"fluide": 0, "moyen": 1, "dense": 2}
    moment_map = {"jour": 0, "nuit": 1}
    vehicule_map = {v: i for i, v in enumerate(VEHICLE_CONFIG.keys())}
    X_multi = np.array([
        [r["distance"],
         trafic_map.get(r["trafic"], 0),
         moment_map.get(r["moment"], 0),
         vehicule_map.get(r["vehicule"], 0)]
        for r in receipts
    ])
    lr_multi = LinearRegression()
    lr_multi.fit(X_multi, tarifs)
    r2_multi = round(lr_multi.score(X_multi, tarifs), 4)

    # -- PCA --
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_multi)
    pca = PCA(n_components=min(2, X_multi.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    variance = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    # -- Classification supervisee --
    labels = ["economique" if r["tarif"] < 5000
              else "standard" if r["tarif"] < 15000
              else "premium" for r in receipts]

    results_classif = {}
    if len(set(labels)) > 1 and len(receipts) >= 6:
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, labels, test_size=0.3, random_state=42)
        classifiers = {
            "Random Forest": RandomForestClassifier(n_estimators=10, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=200),
            "KNN": KNeighborsClassifier(n_neighbors=min(3, len(X_tr))),
            "SVM": SVC(),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=10)
        }
        for name, clf in classifiers.items():
            clf.fit(X_tr, y_tr)
            acc = round(accuracy_score(y_te, clf.predict(X_te)), 4)
            results_classif[name] = acc
    else:
        results_classif = {"info": "Besoin de plus de donnees variees"}

    # -- Correlations --
    correlations = {}
    for key, label in [("distance", "Distance"), ("age", "Age")]:
        vals = np.array([float(r[key]) for r in receipts])
        if np.std(vals) > 0:
            corr = round(float(np.corrcoef(vals, tarifs)[0, 1]), 4)
            correlations[label] = corr

    return jsonify({
        "regression_simple": {
            "coef": coef,
            "intercept": intercept,
            "r2": r2_simple,
            "equation": f"Tarif = {coef} x Distance + {intercept}"
        },
        "regression_multiple": {
            "r2": r2_multi,
            "coefs": [round(float(c), 2) for c in lr_multi.coef_]
        },
        "pca": {
            "variance_expliquee": variance,
            "total": round(sum(variance), 4)
        },
        "classification": results_classif,
        "correlations": correlations,
        "nb_recus": len(receipts),
        "chart_data": {
            "distances": [r["distance"] for r in receipts],
            "tarifs": [r["tarif"] for r in receipts],
            "predictions_simple": lr.predict(distances).tolist(),
            "predictions_multiple": lr_multi.predict(X_multi).tolist()
        }
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
