"""Evaluate clearance of 100 static targets with the standing MuJoCo robot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from integrated_shooting_chain.mujoco_sim.salvo import main

if __name__ == '__main__':
    main()
