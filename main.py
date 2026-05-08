"""LatentSeeker training entry point.

Usage:
    python main.py                                                      # dataclass defaults
    python main.py --max_steps 500                                      # defaults + CLI
    python main.py --config_path configs/pretrain.yaml                  # YAML defaults
    python main.py --config_path configs/pretrain.yaml --max_steps 500  # YAML + CLI
"""

from src.training.train import train

if __name__ == "__main__":
    train()
