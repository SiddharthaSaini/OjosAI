import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3

import numpy as np
import cv2
from PIL import Image

import tensorflow as tf
# Use tf.keras via the tensorflow package to avoid static import resolution issues
# Some linters/IDEs cannot resolve the "tensorflow.keras" namespace, so bind load_model via tf.keras
keras = tf.keras
load_model = keras.models.load_model
from werkzeug.security import generate_password_hash, check_password_hash

# ------------------------------------------------
# Flask App Setup
# ------------------------------------------------

app = Flask(__name__)
app.secret_key = "eyecare_ai_secret_key_123"

# Disable oneDNN warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ------------------------------------------------
# Load Model
# ------------------------------------------------

cataract_model = load_model("best_cnn_cataract.h5", compile=False)

# ------------------------------------------------
# Database Setup
# ------------------------------------------------

def get_db_connection():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

def create_users_table():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

create_users_table()

# ------------------------------------------------
# CLAHE Preprocessing (Same as training)
# ------------------------------------------------

def apply_clahe_uint8(img_uint8):
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    return img

def preprocess_clahe_for_cnn(img):
    img = np.array(img)

    if img.max() <= 1.0:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img.astype(np.uint8)

    img_clahe = apply_clahe_uint8(img_uint8)

    img_final = img_clahe.astype(np.float32) / 255.0
    return img_final

# ------------------------------------------------
# Routes
# ------------------------------------------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/diseases_info")
def diseases_info():
    return render_template("diseases_info.html")

# ------------------- AUTH ------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        if user is None:
            flash("User not found. Please sign up first.", "danger")
            return redirect(url_for("login"))

        if not check_password_hash(user["password"], password):
            flash("Incorrect password!", "danger")
            return redirect(url_for("login"))

        # ✅ Session start
        session["logged_in"] = True
        session["user_id"] = user["id"]
        session["user_name"] = user["name"]
        session["user_email"] = user["email"]

        flash("Login successful!", "success")
        return redirect(url_for("prediction"))

    return render_template("login.html")

@app.route("/signup", methods=["POST"])
def signup():
    name = request.form["name"]
    email = request.form["email"]
    password = request.form["password"]

    hashed_password = generate_password_hash(password)

    conn = get_db_connection()
    existing_user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if existing_user:
        conn.close()
        flash("Email already registered. Please login.", "warning")
        return redirect(url_for("login"))

    conn.execute(
        "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
        (name, email, hashed_password)
    )

    conn.commit()
    conn.close()

    flash("Account created successfully. Now login.", "success")
    return redirect(url_for("login", form="signin"))

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully!", "success")
    return redirect(url_for("home"))

# ------------------- PREDICTION ------------------------

@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    if not session.get("logged_in"):
        flash("Please login to access prediction.", "warning")
        return redirect(url_for("login"))

    result = None

    if request.method == "POST":
        file = request.files["retina_image"]

        # Load image
        image = Image.open(file).convert("RGB")
        image = image.resize((224, 224))

        # ✅ Apply same preprocessing as training
        image = preprocess_clahe_for_cnn(image)
        image = np.expand_dims(image, axis=0)

        pred = cataract_model.predict(image, verbose=0)

        prob = float(pred[0][0])

        # 0 = Cataract, 1 = Normal
        if prob < 0.5:
            disease = "Cataract Detected ❗"
            confidence = round((1 - prob) * 100, 2)
        else:
            disease = "Normal Eye ✅"
            confidence = round(prob * 100, 2)

        result = {
            "disease": disease,
            "confidence": f"{confidence}%",
            "advice": "This is an AI-based prediction. Please consult an eye specialist for diagnosis."
        }

    return render_template(
        "prediction.html",
        result=result,
        user_name=session.get("user_name")
    )

# ------------------------------------------------
# Run App
# ------------------------------------------------

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)


