import sqlite3

def init_db():
    conn = sqlite3.connect("transport.db")
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS recus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            numero TEXT,
            nom TEXT,
            prenom TEXT,
            contact TEXT,
            depart TEXT,
            arrivee TEXT,
            distance REAL,
            heure REAL,
            trafic REAL,
            type_vehicule TEXT,
            periode TEXT,
            temps_estime REAL,
            categorie TEXT,
            prix REAL,
            date_creation TEXT
        )
    ''')
    conn.commit()
    conn.close()

def sauvegarder_recu(data):
    conn = sqlite3.connect("transport.db")
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO recus (
            numero, nom, prenom, contact,
            depart, arrivee, distance, heure,
            trafic, type_vehicule, periode,
            temps_estime, categorie, prix, date_creation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data["numero"], data["nom"], data["prenom"],
        data["contact"], data["depart"], data["arrivee"],
        data["distance"], data["heure"], data["trafic"],
        data["type_vehicule"], data["periode"],
        data["temps_estime"], data["categorie"],
        data["prix"], data["date_creation"]
    ))
    conn.commit()
    conn.close()

def get_tous_recus():
    conn = sqlite3.connect("transport.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM recus ORDER BY id DESC")
    recus = cur.fetchall()
    conn.close()
    return recus

def get_recu(id):
    conn = sqlite3.connect("transport.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM recus WHERE id = ?", (id,))
    recu = cur.fetchone()
    conn.close()
    return recu

def supprimer_recu(id):
    conn = sqlite3.connect("transport.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM recus WHERE id = ?", (id,))
    conn.commit()
    conn.close()
