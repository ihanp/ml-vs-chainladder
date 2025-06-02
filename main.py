
import os

print("▶ Running generate_data.py...")
os.system("python generate_data.py")

print("▶ Running prepare_data.py...")
os.system("python prepare_data.py")

print("▶ Running train_model.py...")
os.system("python train_model.py")

print("▶ Running predict_model.py...")
os.system("python predict_model.py")

print("▶ Running chain_ladder.py...")
os.system("python chain_ladder.py")

print("▶ Running plot_results.py...")
os.system("python plot_results.py")

print("✅ Workflow complete.")
