import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate('/Users/rohandavidi/Desktop/treehacks_2024/treehacks_2024/secret_key.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://treehacks2024-1c2ab-default-rtdb.firebaseio.com"
})

ref = db.reference('/')
