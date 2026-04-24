'''Configuration for the Litter Detector application''' 
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
AUTO_RESEARCH_DIR = REPO_ROOT / "auto-research"
sys.path.insert(0, str(AUTO_RESEARCH_DIR))

import train
MODEL_NAME = "models/checkpoints/best_efficientnetb4.pth"
MODEL_CLASS = train.EfficientNetB4UNet
DROPOUT = 0.1

# Zenoh config
ZENOH_ROUTER =  "tcp/localhost:7447"

# OpenTelemetry setup

SERVICE_NAME = "litter-detector"
OTEL_ENDPOINT = "http://127.0.0.1:4317"
