"""
RETRAIN RL AGENT USING REAL USER FEEDBACK REWARDS
This script takes experiences stored from user uploads and retrains the SAC agent

How to use:
    python retrain_rl_agent.py          # Normal retraining
    python retrain_rl_agent.py --force  # Force retraining even with less data
    python retrain_rl_agent.py --timesteps 10000  # Custom timesteps
"""

import numpy as np
import json
import sqlite3
import os
import sys
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces
import random
from datetime import datetime

# ============================================================================
# STEP 1: LOAD STORED EXPERIENCES FROM DATABASE
# ============================================================================

def load_stored_experiences(limit=None):
    """Load experiences stored from user feedback"""
    conn = sqlite3.connect('rl_experiences.db')
    c = conn.cursor()
    
    query = "SELECT session_id, state_before, action, reward, state_after, done, timestamp FROM rl_experiences ORDER BY timestamp DESC"
    if limit:
        query += f" LIMIT {limit}"
    
    c.execute(query)
    rows = c.fetchall()
    conn.close()
    
    experiences = []
    for row in rows:
        experiences.append({
            "session_id": row[0],
            "state_before": json.loads(row[1]),
            "action": row[2],
            "reward": row[3],
            "state_after": json.loads(row[4]),
            "done": bool(row[5]),
            "timestamp": row[6]
        })
    
    print(f"📚 Loaded {len(experiences)} experiences from database")
    return experiences


def load_reward_sessions():
    """Load reward sessions from plant_sessions table"""
    conn = sqlite3.connect('spray_logs.db')
    c = conn.cursor()
    c.execute("""SELECT session_id, plant_breed, disease_name, severity_before, 
                      severity_after, spray_amount, reward, improvement_percent 
                 FROM plant_sessions WHERE reward IS NOT NULL ORDER BY id DESC""")
    rows = c.fetchall()
    conn.close()
    
    sessions = []
    for row in rows:
        sessions.append({
            "session_id": row[0],
            "plant_breed": row[1],
            "disease": row[2],
            "severity_before": row[3],
            "severity_after": row[4],
            "spray_amount": row[5],
            "reward": row[6],
            "improvement": row[7]
        })
    
    print(f"🌱 Loaded {len(sessions)} reward sessions from database")
    return sessions


# ============================================================================
# STEP 2: CREATE ENVIRONMENT THAT USES REAL EXPERIENCES
# ============================================================================

class RealExperienceEnv(gym.Env):
    """
    Environment that trains using REAL user experiences instead of simulation
    """
    
    def __init__(self, experiences):
        super().__init__()
        self.experiences = experiences
        self.current_idx = 0
        
        # State space: [disease_id, severity, temperature, humidity, soil_moisture, time_since_last_spray]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -10, 0, 0, 0], dtype=np.float32),
            high=np.array([50, 1, 50, 100, 100, 168], dtype=np.float32)
        )
        
        # Action space: spray_amount (0-500 ml normalized)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        
        self.current_state = None
        self.expected_action = None
        self.expected_reward = None
        self.expected_next_state = None
        self.done = False
    
    def reset(self, seed=None, options=None):
        """Start a new episode with a random real experience"""
        self.current_idx = random.randint(0, len(self.experiences) - 1)
        exp = self.experiences[self.current_idx]
        
        # Use the state_before as initial state
        self.current_state = np.array(exp["state_before"], dtype=np.float32)
        self.expected_action = exp["action"]
        self.expected_reward = exp["reward"]
        self.expected_next_state = np.array(exp["state_after"], dtype=np.float32)
        self.done = exp["done"]
        
        return self.current_state, {}
    
    def step(self, action):
        """Compare agent's action with the REAL action that was taken"""
        # Calculate how close the agent's action is to the real action
        action_taken = float(action[0])
        real_action = self.expected_action
        
        # Calculate reward based on similarity to real action
        # The closer the action to what actually worked, the higher the reward
        action_similarity = 1 - min(1, abs(action_taken - real_action) / 0.5)
        
        # Use the REAL reward from user feedback as base
        base_reward = self.expected_reward
        
        # Combine: reward is high if agent chooses similar action to what worked
        # This teaches the agent to mimic successful human decisions
        combined_reward = (base_reward * 0.6) + (action_similarity * 0.4)
        
        # Move to next state
        self.current_state = self.expected_next_state
        
        return self.current_state, combined_reward, self.done, False, {
            "action_taken": action_taken,
            "real_action": real_action,
            "base_reward": base_reward,
            "action_similarity": action_similarity,
            "combined_reward": combined_reward
        }


# ============================================================================
# STEP 3: RETRAIN THE SAC AGENT
# ============================================================================

class RetrainCallback(BaseCallback):
    """Callback to save model during retraining"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_reward = -np.inf
        self.episode_rewards = []
    
    def _on_step(self) -> bool:
        return True
    
    def _on_episode_end(self):
        # Track episode reward
        if len(self.model.ep_info_buffer) > 0:
            ep_reward = self.model.ep_info_buffer[-1]['r']
            self.episode_rewards.append(ep_reward)
            
            if self.verbose > 0:
                print(f"📊 Episode {len(self.episode_rewards)}: Reward = {ep_reward:.2f}")
        
        # Save checkpoint every 5000 timesteps
        if self.num_timesteps % 5000 == 0 and self.num_timesteps > 0:
            checkpoint_path = f"crop_sac_model_checkpoint_{self.num_timesteps}"
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"💾 Checkpoint saved: {checkpoint_path}")


def retrain_with_real_experiences(experiences, timesteps=10000, verbose=1):
    """Retrain the SAC agent using real user experiences"""
    
    print("\n" + "="*60)
    print("🔄 RETRAINING SAC AGENT WITH REAL USER FEEDBACK")
    print("="*60)
    
    # Load existing model if available
    model_path = "crop_sac_model.zip"
    if os.path.exists(model_path):
        print(f"📂 Loading existing model from {model_path}")
        model = SAC.load(model_path)
    else:
        print("⚠️ No existing model found. Creating new model...")
        # Create a temporary environment for initialization
        temp_env = RealExperienceEnv(experiences[:10])
        model = SAC("MlpPolicy", temp_env, verbose=0)
    
    # Create environment with real experiences
    print(f"📊 Creating environment with {len(experiences)} real experiences")
    env = RealExperienceEnv(experiences)
    
    # Set lower learning rate for fine-tuning (don't change too much)
    model.learning_rate = 1e-4
    print(f"📈 Learning rate set to {model.learning_rate} for fine-tuning")
    
    print(f"🏋️ Retraining for {timesteps} timesteps...")
    
    # Create callback
    callback = RetrainCallback(verbose=verbose)
    
    # Retrain the model
    model.set_env(env)
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        reset_num_timesteps=False  # Don't reset, continue learning
    )
    
    # Save the retrained model
    model.save("crop_sac_model_retrained")
    print(f"✅ Retrained model saved as 'crop_sac_model_retrained.zip'")
    
    # Also save as best model
    model.save("crop_sac_model_best")
    print(f"✅ Best model saved as 'crop_sac_model_best.zip'")
    
    return model, callback.episode_rewards


# ============================================================================
# STEP 4: EVALUATE THE RETRAINED MODEL
# ============================================================================

def evaluate_model(model, test_experiences, n_episodes=10):
    """Evaluate the retrained model against test experiences"""
    
    print("\n" + "="*60)
    print("📊 EVALUATING RETRAINED MODEL")
    print("="*60)
    
    if len(test_experiences) == 0:
        print("⚠️ No test experiences available")
        return 0
    
    env = RealExperienceEnv(test_experiences)
    total_reward = 0
    action_similarities = []
    
    for episode in range(min(n_episodes, len(test_experiences))):
        obs, _ = env.reset()
        episode_reward = 0
        step = 0
        episode_similarities = []
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            episode_similarities.append(info['action_similarity'])
            step += 1
            
            if done or step > 10:
                break
        
        total_reward += episode_reward
        action_similarities.extend(episode_similarities)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Similarity = {np.mean(episode_similarities):.2f}")
    
    avg_reward = total_reward / n_episodes
    avg_similarity = np.mean(action_similarities) if action_similarities else 0
    
    print(f"\n📈 Evaluation Results:")
    print(f"   Average Reward: {avg_reward:.2f}")
    print(f"   Average Action Similarity: {avg_similarity:.2f}")
    
    return avg_reward, avg_similarity


# ============================================================================
# STEP 5: COMPARE OLD VS NEW MODEL
# ============================================================================

def compare_models(old_model_path, new_model_path, test_experiences):
    """Compare old model vs retrained model"""
    
    print("\n" + "="*60)
    print("🔍 COMPARING OLD VS RETRAINED MODEL")
    print("="*60)
    
    # Load both models
    old_model = None
    if os.path.exists(old_model_path):
        old_model = SAC.load(old_model_path)
        print("✅ Loaded old model")
    else:
        print("⚠️ Old model not found")
    
    new_model = SAC.load(new_model_path)
    print("✅ Loaded retrained model")
    
    if len(test_experiences) == 0:
        print("⚠️ No test experiences for comparison")
        return
    
    env = RealExperienceEnv(test_experiences)
    
    results = {"old": [], "new": []}
    similarity_results = {"old": [], "new": []}
    
    for episode in range(min(10, len(test_experiences))):
        obs, _ = env.reset()
        
        # Test old model
        if old_model:
            old_reward = 0
            old_similarities = []
            obs_copy = obs.copy()
            done = False
            step = 0
            while not done and step < 10:
                action, _ = old_model.predict(obs_copy, deterministic=True)
                obs_copy, reward, done, _, info = env.step(action)
                old_reward += reward
                old_similarities.append(info['action_similarity'])
                step += 1
            results["old"].append(old_reward)
            similarity_results["old"].append(np.mean(old_similarities))
        
        # Test new model
        new_reward = 0
        new_similarities = []
        obs_copy = obs.copy()
        done = False
        step = 0
        while not done and step < 10:
            action, _ = new_model.predict(obs_copy, deterministic=True)
            obs_copy, reward, done, _, info = env.step(action)
            new_reward += reward
            new_similarities.append(info['action_similarity'])
            step += 1
        results["new"].append(new_reward)
        similarity_results["new"].append(np.mean(new_similarities))
        
        old_str = f"{results['old'][-1]:.2f}" if results["old"] else "N/A"
        print(f"Episode {episode+1}: Old Reward={old_str}, New Reward={new_reward:.2f}")
    
    if results["old"]:
        old_avg = sum(results["old"]) / len(results["old"])
        new_avg = sum(results["new"]) / len(results["new"])
        improvement = ((new_avg - old_avg) / old_avg) * 100 if old_avg != 0 else 0
        
        old_sim_avg = sum(similarity_results["old"]) / len(similarity_results["old"])
        new_sim_avg = sum(similarity_results["new"]) / len(similarity_results["new"])
        sim_improvement = ((new_sim_avg - old_sim_avg) / old_sim_avg) * 100 if old_sim_avg != 0 else 0
        
        print(f"\n📊 Summary:")
        print(f"   Old Model Avg Reward: {old_avg:.2f}")
        print(f"   New Model Avg Reward: {new_avg:.2f}")
        print(f"   Reward Improvement: {improvement:+.1f}%")
        print(f"")
        print(f"   Old Model Action Similarity: {old_sim_avg:.2f}")
        print(f"   New Model Action Similarity: {new_sim_avg:.2f}")
        print(f"   Similarity Improvement: {sim_improvement:+.1f}%")
        
        if improvement > 0:
            print(f"\n🎉 Retrained model is BETTER! {improvement:+.1f}% improvement")
        elif improvement < 0:
            print(f"\n⚠️ Retrained model is WORSE. Consider more data or different hyperparameters")
        else:
            print(f"\n📊 Models perform similarly")
    else:
        print(f"\n📊 New Model Avg Reward: {sum(results['new'])/len(results['new']):.2f}")


# ============================================================================
# STEP 6: GENERATE RETRAINING REPORT
# ============================================================================

def generate_report(experiences, episode_rewards, evaluation_score):
    """Generate a report of the retraining process"""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_experiences": len(experiences),
        "rewards_used": [exp["reward"] for exp in experiences],
        "episode_rewards": episode_rewards,
        "evaluation_score": evaluation_score,
        "statistics": {
            "avg_experience_reward": np.mean([exp["reward"] for exp in experiences]),
            "min_experience_reward": min([exp["reward"] for exp in experiences]),
            "max_experience_reward": max([exp["reward"] for exp in experiences]),
            "positive_experiences": len([exp for exp in experiences if exp["reward"] > 0]),
            "negative_experiences": len([exp for exp in experiences if exp["reward"] < 0]),
            "avg_episode_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "best_episode_reward": max(episode_rewards) if episode_rewards else 0
        }
    }
    
    # Save report
    with open('retraining_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to retraining_report.json")
    
    return report


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Retrain SAC agent with user feedback')
    parser.add_argument('--force', action='store_true', help='Force retraining even with less data')
    parser.add_argument('--timesteps', type=int, default=5000, help='Number of timesteps for retraining')
    parser.add_argument('--no-test', action='store_true', help='Skip testing')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level (0-2)')
    args = parser.parse_args()
    
    print("="*60)
    print("🔄 RL AGENT RETRAINING SYSTEM")
    print("="*60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚙️ Arguments: timesteps={args.timesteps}, force={args.force}")
    
    # Step 1: Load experiences from database
    print("\n📥 Step 1: Loading stored experiences...")
    experiences = load_stored_experiences()
    
    # Check if we have enough experiences
    min_required = 10
    if len(experiences) < min_required:
        if args.force:
            print(f"⚠️ Only {len(experiences)} experiences found. Forcing retraining anyway...")
        else:
            print(f"❌ Only {len(experiences)} experiences found. Need at least {min_required} for retraining.")
            print("   Upload more before/after images to collect more data!")
            print("   Use --force to override this check")
            return
    
    # Also load reward sessions for additional info
    reward_sessions = load_reward_sessions()
    
    # Display statistics
    rewards = [exp["reward"] for exp in experiences]
    print(f"\n📈 Reward Statistics from User Feedback:")
    print(f"   Total Experiences: {len(experiences)}")
    print(f"   Average Reward: {sum(rewards)/len(rewards):.2f}")
    print(f"   Best Reward: {max(rewards):.2f}")
    print(f"   Worst Reward: {min(rewards):.2f}")
    print(f"   Positive Rewards: {len([r for r in rewards if r > 0])}")
    print(f"   Negative Rewards: {len([r for r in rewards if r < 0])}")
    print(f"   Reward Sessions: {len(reward_sessions)}")
    
    # Split into train and test sets (80% train, 20% test)
    split_idx = int(len(experiences) * 0.8)
    random.shuffle(experiences)  # Shuffle to avoid bias
    train_experiences = experiences[:split_idx]
    test_experiences = experiences[split_idx:]
    
    print(f"\n📊 Data Split:")
    print(f"   Training set: {len(train_experiences)} experiences")
    print(f"   Test set: {len(test_experiences)} experiences")
    
    # Step 2: Retrain the model
    print("\n🏋️ Step 2: Retraining SAC agent...")
    try:
        retrained_model, episode_rewards = retrain_with_real_experiences(
            train_experiences, 
            timesteps=args.timesteps,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"❌ Retraining failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Evaluate the model
    if not args.no_test and len(test_experiences) > 0:
        print("\n📊 Step 3: Evaluating retrained model...")
        eval_reward, eval_similarity = evaluate_model(retrained_model, test_experiences)
    else:
        eval_reward = 0
        eval_similarity = 0
        print("\n⚠️ Skipping evaluation")
    
    # Step 4: Compare with old model
    if os.path.exists("crop_sac_model.zip") and not args.no_test:
        compare_models("crop_sac_model.zip", "crop_sac_model_retrained.zip", test_experiences)
    
    # Step 5: Generate report
    print("\n📄 Step 5: Generating report...")
    generate_report(experiences, episode_rewards, eval_reward)
    
    # Step 6: Ask if user wants to deploy
    print("\n" + "="*60)
    print("✅ RETRAINING COMPLETE!")
    print("="*60)
    print("\n📁 Models saved:")
    print("   - crop_sac_model_retrained.zip (retrained version)")
    print("   - crop_sac_model_best.zip (best version)")
    print("   - crop_sac_model_checkpoint_*.zip (checkpoints)")
    print("\n💡 To deploy the new model:")
    print("   1. Stop your Flask app")
    print("   2. Rename crop_sac_model_retrained.zip to crop_sac_model.zip")
    print("   3. Restart your Flask app")
    print("\n   OR use the API endpoint: POST /api/deploy_model")
    print("="*60)


# ============================================================================
# DEPLOYMENT HELPER
# ============================================================================

def deploy_retrained_model():
    """Helper function to deploy the retrained model"""
    if os.path.exists("crop_sac_model_retrained.zip"):
        # Backup old model
        if os.path.exists("crop_sac_model.zip"):
            os.rename("crop_sac_model.zip", "crop_sac_model_backup.zip")
            print("📦 Old model backed up as crop_sac_model_backup.zip")
        
        # Deploy new model
        os.rename("crop_sac_model_retrained.zip", "crop_sac_model.zip")
        print("✅ New model deployed as crop_sac_model.zip")
        return True
    else:
        print("❌ No retrained model found. Run retraining first.")
        return False


if __name__ == "__main__":
    # Check if deploy command
    if len(sys.argv) > 1 and sys.argv[1] == "--deploy":
        deploy_retrained_model()
    else:
        main()