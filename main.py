"""
Run the full predictive-maintenance pipeline end to end.
  python main.py
Stages: build dataset -> classical models -> 1D-CNN -> physics plot.
"""
import subprocess, sys
for stage in ["build_dataset.py", "models_classical.py", "cnn.py", "physics_plot.py"]:
    print(f"\n{'='*60}\nRUNNING {stage}\n{'='*60}")
    subprocess.run([sys.executable, stage], check=True)
print("\nDone. See outputs/ for plots, metrics, and saved models.")
