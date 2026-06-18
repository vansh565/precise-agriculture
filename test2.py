"""
PHASAL AI - Complete Plant Disease Detection System with RL Feedback
- CNN Disease Detection
- SAC Reinforcement Learning Agent
- Before/After Reward System
- RL Feedback Buffer for Continuous Learning
- Retraining API
- Complete History with Rewards
"""

from flask import Flask, render_template, request, redirect, send_from_directory, url_for, jsonify, session
import numpy as np
import json
import uuid
import tensorflow as tf
import random
from datetime import datetime
import os
import sqlite3
import subprocess

# Try to import vector DB (optional)
try:
    from vectordb import PlantDiseaseVectorDB
    vector_db_available = True
except ImportError:
    vector_db_available = False
    print("⚠️ vectordb not found. Vector database features disabled.")

app = Flask(__name__)
app.secret_key = 'phasal_ai_secret_key_2024'

# ============================================================================
# INITIALIZE VECTOR DATABASE (if available)
# ============================================================================

if vector_db_available:
    vector_db = PlantDiseaseVectorDB()
else:
    vector_db = None

# ============================================================================
# LOAD CNN MODEL
# ============================================================================

model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

# Labels
label = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

# Load disease database
with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)

disease_lookup = {d['name']: d for d in plant_disease}

# ============================================================================
# RL FEEDBACK SYSTEM
# ============================================================================

class RLFeedbackBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size
        self.db_initialized = False
        self._init_db()
    
    def _init_db(self):
        try:
            conn = sqlite3.connect('rl_experiences.db')
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS rl_experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                state_before TEXT,
                action REAL,
                reward REAL,
                state_after TEXT,
                done INTEGER,
                timestamp TEXT
            )''')
            conn.commit()
            conn.close()
            self.db_initialized = True
            print("✅ RL Feedback Database initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize RL database: {e}")
    
    def add_experience(self, session_id, state_before, action, reward, state_after, done=False):
        experience = {
            "session_id": session_id,
            "state_before": state_before,
            "action": action,
            "reward": reward,
            "state_after": state_after,
            "done": done,
            "timestamp": datetime.now().isoformat()
        }
        
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
        
        if self.db_initialized:
            try:
                conn = sqlite3.connect('rl_experiences.db')
                c = conn.cursor()
                c.execute("""INSERT INTO rl_experiences 
                             (session_id, state_before, action, reward, state_after, done, timestamp)
                             VALUES (?, ?, ?, ?, ?, ?, ?)""",
                          (session_id, json.dumps(state_before), action, reward, 
                           json.dumps(state_after), int(done), datetime.now().isoformat()))
                conn.commit()
                conn.close()
                print(f"📚 Experience stored: Session={session_id}, Reward={reward}")
            except Exception as e:
                print(f"❌ Failed to store experience: {e}")
        return True
    
    def get_statistics(self):
        if not self.buffer:
            return {"total": 0, "avg_reward": 0, "positive": 0, "negative": 0}
        rewards = [exp["reward"] for exp in self.buffer]
        return {
            "total": len(self.buffer),
            "avg_reward": round(sum(rewards) / len(rewards), 2),
            "positive": len([r for r in rewards if r > 0]),
            "negative": len([r for r in rewards if r < 0]),
            "neutral": len([r for r in rewards if r == 0])
        }

feedback_buffer = RLFeedbackBuffer()

# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def init_db():
    conn = sqlite3.connect('spray_logs.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS plants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plant_breed TEXT UNIQUE,
        created_at TEXT,
        total_sessions INTEGER DEFAULT 0,
        total_reward REAL DEFAULT 0
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS plant_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plant_breed TEXT,
        session_id TEXT UNIQUE,
        before_image TEXT,
        after_image TEXT,
        disease_name TEXT,
        severity_before REAL,
        severity_after REAL,
        spray_amount REAL,
        reward REAL,
        improvement_percent REAL,
        timestamp TEXT,
        status TEXT
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS spray_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        disease_name TEXT,
        severity REAL,
        confidence REAL,
        temperature REAL,
        humidity REAL,
        soil_moisture REAL,
        spray_amount REAL,
        image_path TEXT,
        cause TEXT,
        cure TEXT,
        vector_id TEXT,
        outcome TEXT DEFAULT 'pending'
    )''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

init_db()

# ============================================================================
# LOAD SAC MODEL
# ============================================================================

rl_active = False
sac_model = None

try:
    from stable_baselines3 import SAC
    if os.path.exists("crop_sac_model.zip"):
        sac_model = SAC.load("crop_sac_model")
        rl_active = True
        print("✅ SAC Agent loaded successfully!")
    else:
        print("⚠️ SAC model not found. Using fallback mode.")
except Exception as e:
    print(f"⚠️ Could not load SAC model: {e}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_sensor_readings():
    return round(random.uniform(20, 35), 1), round(random.uniform(40, 85), 1), round(random.uniform(35, 75), 1)

def extract_plant_breed(disease_name):
    if '___' in disease_name:
        return disease_name.split('___')[0]
    elif '_' in disease_name:
        return disease_name.split('_')[0]
    else:
        return disease_name.split()[0] if ' ' in disease_name else disease_name

def get_spray_recommendation(disease_name, severity, temperature, humidity, soil_moisture):
    if "background" in disease_name.lower() or "healthy" in disease_name.lower():
        return 0
    
    base_spray = {"blight": 250, "mildew": 200, "rot": 220, "rust": 180, "spot": 170, "scab": 160}
    base = 150
    for key, value in base_spray.items():
        if key in disease_name.lower():
            base = value
            break
    
    weather_factor = 1.3 if temperature > 25 and humidity > 70 else 0.8 if temperature < 15 else 1.0
    spray_amount = base * severity * weather_factor
    return round(min(max(spray_amount, 0), 500))

def get_urgency_level(spray_amount):
    if spray_amount == 0:
        return "✅ HEALTHY - No treatment required"
    elif spray_amount > 200:
        return "🔥 CRITICAL - Immediate action required!"
    elif spray_amount > 100:
        return "⚠️ HIGH - Spray within 24 hours"
    elif spray_amount > 30:
        return "📋 MEDIUM - Monitor and consider treatment"
    else:
        return "💚 LOW - Minimal treatment needed"

def calculate_reward(severity_before, severity_after, spray_amount):
    improvement = severity_before - severity_after
    
    if improvement > 0.6:
        reward = 2.0
        reward_text = "🌟 Excellent! Great improvement!"
        reward_icon = "🌟"
    elif improvement > 0.4:
        reward = 1.5
        reward_text = "👍 Good! Significant improvement!"
        reward_icon = "👍"
    elif improvement > 0.2:
        reward = 1.0
        reward_text = "📊 Fair! Some improvement noticed"
        reward_icon = "📊"
    elif improvement > 0:
        reward = 0.5
        reward_text = "⚠️ Poor! Minimal improvement"
        reward_icon = "⚠️"
    elif improvement == 0:
        reward = 0.0
        reward_text = "📊 No change detected"
        reward_icon = "📊"
    else:
        reward = -1.0
        reward_text = "❌ Bad! Condition worsened"
        reward_icon = "❌"
    
    if spray_amount > 300 and improvement < 0.3:
        reward -= 0.3
        reward_text += " (Penalty: Over-spraying)"
    elif spray_amount < 100 and improvement > 0.5:
        reward += 0.3
        reward_text += " (Bonus: Efficient spraying!)"
    
    return round(max(-1.0, min(2.0, reward)), 2), reward_text, reward_icon

def register_plant(plant_breed):
    conn = sqlite3.connect('spray_logs.db')
    c = conn.cursor()
    try:
        c.execute("INSERT OR IGNORE INTO plants (plant_breed, created_at, total_sessions, total_reward) VALUES (?, ?, 0, 0)",
                  (plant_breed, datetime.now().isoformat()))
        conn.commit()
    except Exception as e:
        print(f"Error registering plant: {e}")
    finally:
        conn.close()

def get_disease_id(disease_name):
    for idx, name in enumerate(label):
        if name == disease_name:
            return idx
    return 0

# ============================================================================
# PLANT HISTORY FUNCTIONS
# ============================================================================

def get_plant_history(plant_breed):
    conn = sqlite3.connect('spray_logs.db')
    c = conn.cursor()
    try:
        c.execute("""SELECT session_id, disease_name, severity_before, severity_after, reward, improvement_percent, timestamp 
                     FROM plant_sessions WHERE plant_breed = ? ORDER BY id DESC""", (plant_breed,))
        rows = c.fetchall()
        sessions = []
        for row in rows:
            sessions.append({
                "session_id": row[0],
                "disease": row[1],
                "severity_before": row[2],
                "severity_after": row[3],
                "reward": row[4] if row[4] else None,
                "improvement": row[5],
                "date": row[6]
            })
        return sessions
    except Exception as e:
        print(f"Error getting plant history: {e}")
        return []
    finally:
        conn.close()

def get_all_plants():
    conn = sqlite3.connect('spray_logs.db')
    c = conn.cursor()
    try:
        c.execute("SELECT plant_breed, created_at, total_sessions, total_reward FROM plants ORDER BY total_reward DESC")
        rows = c.fetchall()
        plants = []
        for row in rows:
            plants.append({
                "name": row[0],
                "created_at": row[1],
                "sessions": row[2],
                "total_reward": round(row[3], 2) if row[3] else 0
            })
        return plants
    except Exception as e:
        print(f"Error getting plants: {e}")
        return []
    finally:
        conn.close()

def get_reward_statistics():
    conn = sqlite3.connect('spray_logs.db')
    c = conn.cursor()
    try:
        c.execute("SELECT AVG(reward), MAX(reward), MIN(reward), COUNT(*) FROM plant_sessions WHERE reward IS NOT NULL")
        avg_reward, max_reward, min_reward, total = c.fetchone()
        
        c.execute("SELECT COUNT(*) FROM plant_sessions WHERE reward >= 1.5")
        excellent = c.fetchone()[0] or 0
        c.execute("SELECT COUNT(*) FROM plant_sessions WHERE reward >= 1.0 AND reward < 1.5")
        good = c.fetchone()[0] or 0
        c.execute("SELECT COUNT(*) FROM plant_sessions WHERE reward >= 0.5 AND reward < 1.0")
        fair = c.fetchone()[0] or 0
        c.execute("SELECT COUNT(*) FROM plant_sessions WHERE reward < 0.5 AND reward >= 0")
        poor = c.fetchone()[0] or 0
        c.execute("SELECT COUNT(*) FROM plant_sessions WHERE reward < 0")
        bad = c.fetchone()[0] or 0
        
        return {
            "average_reward": round(avg_reward, 2) if avg_reward else 0,
            "best_reward": round(max_reward, 2) if max_reward else 0,
            "worst_reward": round(min_reward, 2) if min_reward else 0,
            "total_sessions": total or 0,
            "excellent": excellent,
            "good": good,
            "fair": fair,
            "poor": poor,
            "bad": bad
        }
    except Exception as e:
        return {"average_reward": 0, "best_reward": 0, "worst_reward": 0, "total_sessions": 0,
                "excellent": 0, "good": 0, "fair": 0, "poor": 0, "bad": 0}
    finally:
        conn.close()

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def home():
    return render_template('home.html', result=False)

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160, 160))
    return np.array([tf.keras.utils.img_to_array(image)])

def predict_disease(image_path):
    img = extract_features(image_path)
    predictions = model.predict(img, verbose=0)
    predicted_idx = predictions.argmax()
    predicted_label = label[predicted_idx]
    confidence = float(np.max(predictions[0]))
    severity = 0.05 if "healthy" in predicted_label.lower() else confidence
    return predicted_label, severity, confidence

@app.route('/upload/', methods=['POST'])
def uploadimage():
    if request.method == "POST":
        try:
            if not os.path.exists('uploadimages'):
                os.makedirs('uploadimages')
            
            image = request.files['img']
            if not image:
                return redirect('/')
            
            filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
            image_path = os.path.join('uploadimages', filename)
            image.save(image_path)
            
            predicted_label, severity, confidence = predict_disease(image_path)
            plant_breed = extract_plant_breed(predicted_label)
            
            disease_data = disease_lookup.get(predicted_label, {})
            cause = disease_data.get("cause", "Information not available")
            cure = disease_data.get("cure", "Consult agricultural expert")
            temp, humidity, soil_moisture = get_sensor_readings()
            spray_amount = get_spray_recommendation(predicted_label, severity, temp, humidity, soil_moisture)
            urgency = get_urgency_level(spray_amount)
            
            register_plant(plant_breed)
            
            vector_insights = {"similar_cases_found": 0}
            if vector_db:
                record = {
                    "disease_name": predicted_label,
                    "severity": severity,
                    "temperature": temp,
                    "humidity": humidity,
                    "soil_moisture": soil_moisture,
                    "spray_amount": spray_amount,
                    "timestamp": datetime.now().isoformat(),
                    "confidence": confidence,
                    "outcome": "pending"
                }
                similar_cases = vector_db.find_similar_conditions(record, n_results=5)
                vector_recommendation = vector_db.get_recommendation_from_similar(record)
                vector_id = vector_db.add_record(record)
                vector_insights = {
                    "similar_cases_found": len(similar_cases),
                    "similarity_score": similar_cases[0]['distance'] if similar_cases else 0,
                    "vector_recommendation": vector_recommendation
                }
            else:
                vector_id = None
            
            conn = sqlite3.connect('spray_logs.db')
            c = conn.cursor()
            c.execute("""INSERT INTO spray_logs 
                         (timestamp, disease_name, severity, confidence, temperature, 
                          humidity, soil_moisture, spray_amount, image_path, cause, cure, vector_id, outcome)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                      (datetime.now().isoformat(), predicted_label, severity, confidence, 
                       temp, humidity, soil_moisture, spray_amount, image_path, cause, cure, vector_id, "pending"))
            conn.commit()
            conn.close()
            
            prediction_data = {
                "name": predicted_label.replace("_", " ").replace("___", " - "),
                "plant_breed": plant_breed,
                "cause": cause,
                "cure": cure,
                "confidence": f"{confidence:.1%}",
                "severity": f"{severity:.1%}",
                "severity_value": severity,
                "temperature": f"{temp:.1f}°C",
                "humidity": f"{humidity:.1f}%",
                "soil_moisture": f"{soil_moisture:.1f}%",
                "spray_amount": f"{spray_amount} ml",
                "spray_value": spray_amount,
                "urgency": urgency,
                "rl_active": rl_active,
                "vector_insights": vector_insights
            }
            
            return render_template('home.html', result=True, imagepath=f'/uploadimages/{filename}', prediction=prediction_data)
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return render_template('home.html', result=False, error=str(e))
    
    return redirect('/')

@app.route('/upload_before/', methods=['POST'])
def upload_before():
    if 'before_image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['before_image']
    plant_breed = request.form.get('plant_breed', 'Unknown')
    
    os.makedirs('static/uploads', exist_ok=True)
    session_id = str(uuid.uuid4())
    filename = f"before_{session_id}_{file.filename}"
    filepath = os.path.join('static/uploads', filename)
    file.save(filepath)
    
    disease_name, severity, confidence = predict_disease(filepath)
    temp, humidity, soil = get_sensor_readings()
    spray_amount = get_spray_recommendation(disease_name, severity, temp, humidity, soil)
    
    register_plant(plant_breed)
    
    conn = sqlite3.connect('spray_logs.db')
    c = conn.cursor()
    c.execute("""INSERT INTO plant_sessions 
                 (plant_breed, session_id, before_image, disease_name, severity_before, spray_amount, timestamp, status)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
              (plant_breed, session_id, filepath, disease_name, severity, spray_amount, 
               datetime.now().isoformat(), "sprayed"))
    conn.commit()
    conn.close()
    
    return jsonify({
        "session_id": session_id,
        "plant_breed": plant_breed,
        "disease": disease_name.replace("_", " ").replace("___", " - "),
        "severity": f"{severity:.1%}",
        "severity_value": severity,
        "confidence": f"{confidence:.1%}",
        "spray_amount": f"{spray_amount} ml",
        "spray_value": spray_amount,
        "urgency": get_urgency_level(spray_amount),
        "message": f"✅ Session started for {plant_breed}! Upload after image in 7 days to get reward!"
    })

@app.route('/upload_after/', methods=['POST'])
def upload_after():
    if 'after_image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['after_image']
    session_id = request.form.get('session_id')
    
    if not session_id:
        return jsonify({"error": "No session ID"}), 400
    
    conn = sqlite3.connect('spray_logs.db')
    c = conn.cursor()
    c.execute("SELECT plant_breed, severity_before, spray_amount, disease_name FROM plant_sessions WHERE session_id = ?", (session_id,))
    row = c.fetchone()
    
    if not row:
        conn.close()
        return jsonify({"error": "Session not found"}), 404
    
    plant_breed, severity_before, spray_amount, disease_before = row
    
    filename = f"after_{session_id}_{file.filename}"
    filepath = os.path.join('static/uploads', filename)
    file.save(filepath)
    
    disease_after, severity_after, confidence = predict_disease(filepath)
    
    # Calculate reward
    reward, reward_text, reward_icon = calculate_reward(severity_before, severity_after, spray_amount)
    improvement_percent = round((severity_before - severity_after) * 100, 1)
    improved = severity_after < severity_before
    
    # Store experience for RL
    temp, humidity, soil = get_sensor_readings()
    
    state_before = [
        get_disease_id(disease_before),
        severity_before,
        temp,
        humidity,
        soil,
        0
    ]
    
    state_after = [
        get_disease_id(disease_after),
        severity_after,
        temp + random.uniform(-1, 1),
        humidity + random.uniform(-2, 2),
        max(0, soil - (spray_amount / 500) * 10),
        0
    ]
    
    done = severity_after < 0.1
    feedback_buffer.add_experience(
        session_id=session_id,
        state_before=state_before,
        action=spray_amount / 500.0,
        reward=reward,
        state_after=state_after,
        done=done
    )
    
    # Update database
    c.execute("""UPDATE plant_sessions 
                 SET after_image = ?, severity_after = ?, reward = ?, improvement_percent = ?, status = ?
                 WHERE session_id = ?""",
              (filepath, severity_after, reward, improvement_percent, "completed", session_id))
    
    c.execute("UPDATE plants SET total_sessions = total_sessions + 1, total_reward = total_reward + ? WHERE plant_breed = ?", (reward, plant_breed))
    conn.commit()
    conn.close()
    
    if reward >= 1.5:
        reward_level = "excellent"
    elif reward >= 1.0:
        reward_level = "good"
    elif reward >= 0.5:
        reward_level = "fair"
    elif reward >= 0:
        reward_level = "poor"
    else:
        reward_level = "bad"
    
    return jsonify({
        "session_id": session_id,
        "plant_breed": plant_breed,
        "disease_before": disease_before.replace("_", " ").replace("___", " - "),
        "disease_after": disease_after.replace("_", " ").replace("___", " - "),
        "severity_before": f"{severity_before:.1%}",
        "severity_after": f"{severity_after:.1%}",
        "improvement": f"{improvement_percent}%",
        "improved": improved,
        "spray_amount": f"{spray_amount} ml",
        "reward": reward,
        "reward_level": reward_level,
        "reward_icon": reward_icon,
        "reward_text": reward_text,
        "rl_learning": "✅ This reward will help the AI learn and improve future recommendations!",
        "message": f"🎉 REWARD: {reward} points! {reward_text}"
    })

# ============================================================================
# API ROUTES - INCLUDING THE MISSING ONE!
# ============================================================================

@app.route('/api/reward_sessions')
def get_reward_sessions():
    """Get all reward sessions (before/after tracking) - THIS WAS MISSING!"""
    conn = sqlite3.connect('spray_logs.db')
    c = conn.cursor()
    try:
        c.execute("""SELECT session_id, plant_breed, disease_name, severity_before, 
                          severity_after, spray_amount, reward, improvement_percent, 
                          status, timestamp 
                     FROM plant_sessions ORDER BY id DESC LIMIT 100""")
        rows = c.fetchall()
        conn.close()
        
        sessions = []
        for row in rows:
            sessions.append({
                "session_id": row[0],
                "plant_breed": row[1] if row[1] else "Unknown",
                "disease": row[2],
                "severity_before": row[3],
                "severity_after": row[4],
                "spray_amount": row[5],
                "reward": row[6],
                "improvement": row[7],
                "status": row[8] if row[8] else "pending",
                "date": row[9]
            })
        return jsonify(sessions)
    except Exception as e:
        print(f"Error getting reward sessions: {e}")
        return jsonify([])

@app.route('/api/realtime/plants')
def realtime_plants():
    conn = sqlite3.connect('spray_logs.db')
    c = conn.cursor()
    c.execute("SELECT id, plant_breed, created_at, total_sessions, total_reward FROM plants ORDER BY total_reward DESC")
    rows = c.fetchall()
    conn.close()
    
    plants = []
    for row in rows:
        plants.append({
            "id": row[0],
            "name": row[1],
            "created_at": row[2],
            "sessions": row[3],
            "total_reward": row[4]
        })
    
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "count": len(plants),
        "data": plants
    })

@app.route('/api/realtime/sessions')
def realtime_sessions():
    conn = sqlite3.connect('spray_logs.db')
    c = conn.cursor()
    c.execute("""SELECT session_id, plant_breed, disease_name, 
                      severity_before, severity_after, spray_amount, reward, 
                      improvement_percent, status, timestamp 
                 FROM plant_sessions ORDER BY id DESC LIMIT 20""")
    rows = c.fetchall()
    conn.close()
    
    sessions = []
    for row in rows:
        sessions.append({
            "session_id": row[0][:8] if row[0] else "N/A",
            "plant_breed": row[1] if row[1] else "Unknown",
            "disease": row[2],
            "severity_before": round(row[3] * 100, 1) if row[3] else 0,
            "severity_after": round(row[4] * 100, 1) if row[4] else 0,
            "spray_amount": row[5],
            "reward": row[6],
            "improvement": row[7],
            "status": row[8] if row[8] else "pending",
            "timestamp": row[9][:19] if row[9] else ""
        })
    
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "count": len(sessions),
        "data": sessions
    })

@app.route('/api/realtime/experiences')
def realtime_experiences():
    if not os.path.exists('rl_experiences.db'):
        return jsonify({"timestamp": datetime.now().isoformat(), "count": 0, "data": []})
    
    conn = sqlite3.connect('rl_experiences.db')
    c = conn.cursor()
    c.execute("SELECT session_id, action, reward, done, timestamp FROM rl_experiences ORDER BY id DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()
    
    experiences = []
    for row in rows:
        experiences.append({
            "session_id": row[0][:8] if row[0] else "N/A",
            "action": round(row[1], 3) if row[1] else 0,
            "reward": round(row[2], 2) if row[2] else 0,
            "done": bool(row[3]),
            "timestamp": row[4][:19] if row[4] else ""
        })
    
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "count": len(experiences),
        "data": experiences
    })

@app.route('/api/realtime/stats')
def realtime_stats():
    conn = sqlite3.connect('spray_logs.db')
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM plants")
    total_plants = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM plant_sessions")
    total_sessions = c.fetchone()[0]
    
    c.execute("SELECT AVG(reward) FROM plant_sessions WHERE reward IS NOT NULL")
    avg_reward = c.fetchone()[0] or 0
    
    c.execute("SELECT SUM(spray_amount) FROM plant_sessions")
    total_spray = c.fetchone()[0] or 0
    
    c.execute("SELECT COUNT(*) FROM plant_sessions WHERE status='completed'")
    completed = c.fetchone()[0]
    
    conn.close()
    
    rl_count = 0
    if os.path.exists('rl_experiences.db'):
        conn = sqlite3.connect('rl_experiences.db')
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM rl_experiences")
        rl_count = c.fetchone()[0]
        conn.close()
    
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "plants": {"total": total_plants},
        "sessions": {"total": total_sessions, "completed": completed, "pending": total_sessions - completed},
        "rewards": {"average": round(avg_reward, 2), "total_spray_ml": total_spray},
        "rl": {"experiences": rl_count, "ready_for_retraining": rl_count >= 10}
    })

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/plant_history/<plant_breed>')
def api_plant_history(plant_breed):
    sessions = get_plant_history(plant_breed)
    return jsonify(sessions)

@app.route('/api/all_plants')
def api_all_plants():
    plants = get_all_plants()
    return jsonify(plants)

@app.route('/api/reward_stats')
def api_reward_stats():
    stats = get_reward_statistics()
    rl_stats = feedback_buffer.get_statistics()
    stats["rl_feedback"] = rl_stats
    return jsonify(stats)

@app.route('/api/rl_feedback_stats')
def api_rl_feedback_stats():
    stats = feedback_buffer.get_statistics()
    return jsonify(stats)

@app.route('/api/vector_stats')
def api_vector_stats():
    if vector_db:
        stats = vector_db.get_statistics()
        trends = vector_db.find_disease_trends()
        return jsonify({"stats": stats, "trends": trends})
    return jsonify({"available": False})

@app.route('/api/spray_history')
def get_spray_history():
    try:
        conn = sqlite3.connect('spray_logs.db')
        c = conn.cursor()
        c.execute("SELECT id, timestamp, disease_name, severity, temperature, humidity, soil_moisture, spray_amount, outcome FROM spray_logs ORDER BY id DESC LIMIT 100")
        rows = c.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                "id": row[0],
                "timestamp": row[1],
                "disease_name": row[2],
                "severity": row[3],
                "temperature": f"{row[4]}°C" if row[4] else "--",
                "humidity": f"{row[5]}%" if row[5] else "--",
                "soil_moisture": f"{row[6]}%" if row[6] else "--",
                "spray_amount": row[7] if row[7] else 0,
                "outcome": row[8] if len(row) > 8 and row[8] else "pending"
            })
        return jsonify(history)
    except Exception as e:
        print(f"Error in spray_history: {e}")
        return jsonify([])

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/api/activate_spray', methods=['POST'])
def activate_spray():
    data = request.json
    print(f"💦 Spray activated: {data.get('amount')}")
    return {"status": "success"}

@app.route('/api/status')
def api_status():
    return {
        "status": "online", 
        "rl_active": rl_active,
        "rl_feedback_count": len(feedback_buffer.buffer),
        "vector_db": vector_db is not None
    }

@app.route('/api/retrain', methods=['POST'])
def trigger_retraining():
    try:
        stats = feedback_buffer.get_statistics()
        if stats['total'] < 10:
            return jsonify({"status": "error", "message": f"Not enough experiences. Need at least 10, have {stats['total']}"}), 400
        
        result = subprocess.run(["python", "retrain_rl_agent.py"], capture_output=True, text=True, timeout=300)
        return jsonify({
            "status": "success",
            "message": "Retraining completed",
            "output": result.stdout[-1000:],
            "model_updated": os.path.exists("crop_sac_model_retrained.zip")
        })
    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "message": "Retraining timeout"}), 408
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/retraining_status')
def retraining_status():
    stats = feedback_buffer.get_statistics()
    return jsonify({
        "total_experiences": stats['total'],
        "average_reward": stats['avg_reward'],
        "positive_rewards": stats['positive'],
        "negative_rewards": stats['negative'],
        "ready_for_retraining": stats['total'] >= 10,
        "model_exists": os.path.exists("crop_sac_model.zip")
    })

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🌿 PHASAL AI - Complete Plant Disease Detection System")
    print("="*60)
    print("✅ CNN Model Loaded")
    print(f"✅ SAC Agent: {'Active' if rl_active else 'Fallback Mode'}")
    print(f"✅ RL Feedback Buffer: Ready")
    print(f"✅ Vector DB: {'Available' if vector_db else 'Not Available'}")
    print("="*60)
    print("📌 Features:")
    print("   - Single Image Detection")
    print("   - Before/After Reward System")
    print("   - RL Feedback Collection")
    print("   - Retraining API (/api/retrain)")
    print("   - Complete History with Rewards")
    print("="*60)
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)