import sqlite3
from app.main import Transaction

conn = sqlite3.connect("fraud_results.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        step INTEGER,
        type TEXT,
        amount REAL,
        nameOrig TEXT,
        oldbalanceOrg REAL,
        newbalanceOrig REAL,
        nameDest TEXT,
        oldbalanceDest REAL,
        newbalanceDest REAL,
        result TEXT
    )
''')
conn.commit()

def save_prediction(data: Transaction, result: str):
    cursor.execute('''
        INSERT INTO predictions (step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, 
            nameDest, oldbalanceDest, newbalanceDest, result)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data.step, data.type, data.amount, data.nameOrig, data.oldbalanceOrg,
        data.newbalanceOrig, data.nameDest, data.oldbalanceDest, data.newbalanceDest, result
    ))
    conn.commit()
