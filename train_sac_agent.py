import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import SAC
import json
import random
import os

class CropSprayEnv(gym.Env):
    """Custom Environment for crop spraying with SAC"""
    
    def __init__(self, disease_db_path="plant_disease.json"):
        super().__init__()
        
        # Load disease database
        try:
            with open(disease_db_path, 'r') as f:
                self.disease_db = json.load(f)
            print(f"✅ Loaded {len(self.disease_db)} diseases from {disease_db_path}")
            
            # Add IDs to diseases if they don't have them
            for idx, disease in enumerate(self.disease_db):
                if 'id' not in disease:
                    disease['id'] = idx
                if 'base_spray_ml' not in disease:
                    # Set default based on disease type
                    if 'healthy' in disease['name'].lower():
                        disease['base_spray_ml'] = 0
                    elif 'blight' in disease['name'].lower():
                        disease['base_spray_ml'] = 250
                    elif 'mildew' in disease['name'].lower():
                        disease['base_spray_ml'] = 200
                    else:
                        disease['base_spray_ml'] = 150
                if 'severity_scaling' not in disease:
                    if 'healthy' in disease['name'].lower():
                        disease['severity_scaling'] = 0
                    elif 'blight' in disease['name'].lower():
                        disease['severity_scaling'] = 1.5
                    else:
                        disease['severity_scaling'] = 1.0
                if 'sensor_thresholds' not in disease:
                    disease['sensor_thresholds'] = {
                        "temperature": {"min": 10, "max": 35, "optimal": 22},
                        "humidity": {"min": 40, "max": 90, "optimal": 65},
                        "soil_moisture": {"min": 40, "max": 85, "optimal": 60}
                    }
                    
        except FileNotFoundError:
            print(f"Error: {disease_db_path} not found!")
            print("Creating default database...")
            self.disease_db = self.create_default_db()
        
        # State: [disease_id, severity, temperature, humidity, soil_moisture, time_since_last_spray_hours]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -10, 0, 0, 0], dtype=np.float32),
            high=np.array([len(self.disease_db), 1, 50, 100, 100, 168], dtype=np.float32)
        )
        
        # Action: spray_amount (0-500 ml)
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32
        )
        
        self.step_count = 0
    
    def create_default_db(self):
        """Create default database if file doesn't exist"""
        return [
            {"id": 0, "name": "Tomato___Late_blight", "base_spray_ml": 250, "severity_scaling": 1.5,
             "sensor_thresholds": {"temperature": {"min": 10, "max": 22, "optimal": 16},
                                  "humidity": {"min": 85, "max": 100, "optimal": 90},
                                  "soil_moisture": {"min": 60, "max": 85, "optimal": 70}}},
            {"id": 1, "name": "Tomato___Early_blight", "base_spray_ml": 200, "severity_scaling": 1.3,
             "sensor_thresholds": {"temperature": {"min": 18, "max": 30, "optimal": 24},
                                  "humidity": {"min": 70, "max": 90, "optimal": 80},
                                  "soil_moisture": {"min": 55, "max": 80, "optimal": 65}}},
            {"id": 2, "name": "Apple___Apple_scab", "base_spray_ml": 150, "severity_scaling": 1.2,
             "sensor_thresholds": {"temperature": {"min": 12, "max": 22, "optimal": 18},
                                  "humidity": {"min": 70, "max": 95, "optimal": 85},
                                  "soil_moisture": {"min": 60, "max": 85, "optimal": 70}}},
            {"id": 3, "name": "Apple___healthy", "base_spray_ml": 0, "severity_scaling": 0,
             "sensor_thresholds": {"temperature": {"min": 15, "max": 28, "optimal": 22},
                                  "humidity": {"min": 40, "max": 70, "optimal": 55},
                                  "soil_moisture": {"min": 50, "max": 70, "optimal": 60}}}
        ]
    
    def calculate_weather_risk(self, temp, humidity, thresholds):
        """Calculate disease spread risk based on weather"""
        temp_risk = 0
        if thresholds["temperature"]["min"] <= temp <= thresholds["temperature"]["max"]:
            temp_range = thresholds["temperature"]["max"] - thresholds["temperature"]["min"]
            if temp_range > 0:
                temp_risk = 1 - abs(temp - thresholds["temperature"]["optimal"]) / temp_range
        
        humidity_risk = 0
        if thresholds["humidity"]["min"] <= humidity <= thresholds["humidity"]["max"]:
            humidity_range = thresholds["humidity"]["max"] - thresholds["humidity"]["min"]
            if humidity_range > 0:
                humidity_risk = 1 - abs(humidity - thresholds["humidity"]["optimal"]) / humidity_range
        
        return (temp_risk + humidity_risk) / 2
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Random initial conditions - only choose from diseased plants (not healthy ones)
        diseased_diseases = [d for d in self.disease_db if 'healthy' not in d['name'].lower()]
        if not diseased_diseases:
            diseased_diseases = self.disease_db
            
        disease_entry = random.choice(diseased_diseases)
        
        self.state = np.array([
            float(disease_entry["id"]),                    # disease_id
            np.random.uniform(0.3, 0.95),                  # severity (starting with disease)
            np.random.uniform(15, 35),                     # temperature (C)
            np.random.uniform(40, 90),                     # humidity (%)
            np.random.uniform(35, 75),                     # soil_moisture (%)
            0                                               # hours since last spray
        ], dtype=np.float32)
        
        self.step_count = 0
        self.disease_info = disease_entry
        
        return self.state, {}
    
    def step(self, action):
        # Convert action (0-1) to spray amount (0-500 ml)
        spray_ml = float(action[0] * 500)
        
        disease_id = int(self.state[0])
        severity = self.state[1]
        temp = self.state[2]
        humidity = self.state[3]
        soil_moisture = self.state[4]
        time_since_spray = self.state[5]
        
        # Get disease info
        disease_info = next((d for d in self.disease_db if d["id"] == disease_id), self.disease_db[0])
        
        # Get thresholds safely
        thresholds = disease_info.get("sensor_thresholds", {
            "temperature": {"min": 10, "max": 30, "optimal": 20},
            "humidity": {"min": 40, "max": 80, "optimal": 60},
            "soil_moisture": {"min": 40, "max": 80, "optimal": 60}
        })
        
        # Calculate weather risk
        weather_risk = self.calculate_weather_risk(temp, humidity, thresholds)
        
        # Calculate optimal spray based on severity and weather
        base_spray = disease_info.get("base_spray_ml", 150)
        severity_scaling = disease_info.get("severity_scaling", 1.0)
        
        if base_spray == 0:  # Healthy plant
            optimal_spray = 0
        else:
            optimal_spray = base_spray * severity * (1 + weather_risk) * severity_scaling
            optimal_spray = min(max(optimal_spray, 0), 500)
        
        # Reward calculation
        if optimal_spray == 0:
            # For healthy plants, reward not spraying
            if spray_ml < 10:
                reward = 1.0
            else:
                reward = -0.5  # Penalty for spraying healthy plants
        else:
            # 1. Disease reduction reward
            if optimal_spray > 0:
                spray_efficiency = 1 - min(1, abs(spray_ml - optimal_spray) / optimal_spray)
            else:
                spray_efficiency = 1 if spray_ml == 0 else 0
            
            disease_reduction = severity * spray_efficiency * 2
            
            # 2. Pesticide waste penalty
            waste_penalty = abs(spray_ml - optimal_spray) / 500 * 0.5
            
            # 3. Environmental penalty for over-spraying
            env_penalty = max(0, (spray_ml - optimal_spray) / 500) * 0.3
            
            # 4. Time efficiency (don't spray too frequently)
            time_penalty = 0
            if time_since_spray < 24 and spray_ml > 50:
                time_penalty = 0.2
            
            # Total reward
            reward = disease_reduction - waste_penalty - env_penalty - time_penalty
            reward = float(np.clip(reward, -1, 2))
        
        # Update state for next step
        if optimal_spray == 0 or spray_ml < 10:
            # No effective spray, disease might worsen
            if weather_risk > 0.7:
                new_severity = min(1, severity + 0.05)
            else:
                new_severity = severity
        else:
            # Disease improves with effective spray
            spray_efficiency = 1 - min(1, abs(spray_ml - optimal_spray) / optimal_spray) if optimal_spray > 0 else 0
            improvement = (spray_ml / 500) * 0.3 * spray_efficiency
            new_severity = max(0, severity - improvement)
            
            # Disease can worsen if conditions are favorable but spray was insufficient
            if weather_risk > 0.7 and spray_ml < optimal_spray * 0.5:
                new_severity = min(1, new_severity + 0.05)
        
        # Update time since last spray
        if spray_ml > 10:
            new_time_since_spray = 0
        else:
            new_time_since_spray = min(168, time_since_spray + 1)
        
        # Update state with noise
        self.state = np.array([
            disease_id,
            new_severity,
            np.clip(temp + np.random.normal(0, 1), -10, 50),
            np.clip(humidity + np.random.normal(0, 2), 0, 100),
            np.clip(soil_moisture - (spray_ml / 500) * 10, 0, 100),
            new_time_since_spray
        ], dtype=np.float32)
        
        self.step_count += 1
        
        # Episode ends after 20 steps or if disease is controlled
        terminated = (new_severity < 0.1) or (self.step_count >= 20)
        truncated = False
        
        info = {
            "spray_ml": spray_ml,
            "optimal_spray": optimal_spray,
            "weather_risk": weather_risk,
            "severity": severity,
            "new_severity": new_severity,
            "reward": reward
        }
        
        return self.state, reward, terminated, truncated, info


# Train the SAC agent
if __name__ == "__main__":
    print("="*60)
    print("CROP SPRAYING RL AGENT TRAINING")
    print("="*60)
    
    print("\n1. Creating environment...")
    env = CropSprayEnv()
    
    print("\n2. Initializing SAC agent...")
    # Disable tensorboard to avoid import error
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        verbose=1
    )
    
    print("\n3. Training SAC agent for 30000 timesteps...")
    print("   (This may take 2-3 minutes)")
    
    try:
        model.learn(total_timesteps=30000, progress_bar=True)
        print("\n✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        print("\nTrying with 10000 timesteps...")
        model.learn(total_timesteps=10000)
        print("✅ Training completed with 10000 timesteps!")
    
    print("\n4. Saving model...")
    model.save("crop_sac_model")
    print("✅ Model saved as 'crop_sac_model.zip'")
    
    print("\n5. Testing trained agent...")
    print("-"*50)
    obs, _ = env.reset()
    total_reward = 0
    total_spray = 0
    
    for i in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_spray += info['spray_ml']
        
        print(f"Step {i+1}:")
        print(f"  💊 Spray: {info['spray_ml']:.1f}ml (optimal: {info['optimal_spray']:.1f}ml)")
        print(f"  🦠 Severity: {info['severity']:.2f} → {info['new_severity']:.2f}")
        print(f"  🎁 Reward: {reward:.2f}")
        print()
        
        if terminated:
            print(f"✅ Episode finished after {i+1} steps!")
            break
    else:
        print(f"✅ Episode complete after 10 steps!")
    
    print("-"*50)
    print(f"📊 Summary:")
    print(f"   Total Reward: {total_reward:.2f}")
    print(f"   Average Spray: {total_spray/10:.1f}ml")
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE! You can now use the model in your Flask app.")
    print("="*60)