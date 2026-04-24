'''Configuration for the Litter Detector application''' 
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
AUTO_RESEARCH_DIR = REPO_ROOT / "auto-research"
sys.path.insert(0, str(AUTO_RESEARCH_DIR))
from train import *
from dataclasses import dataclass
@dataclass
class Settings:
# Model
    MODEL_NAME:str = "models/checkpoints/best_efficientnetb4.pth"
    MODEL_CLASS = EfficientNetB4UNet
    DROPOUT:float = 0.1
    FRAME_MAX_AGE_SECONDS:int = 1
    PROCESSING_TIMEOUT_SECONDS:int = 5
    THRESHOLD:float = 0.8
    
    # Zenoh config
    ZENOH_ROUTER:str =  "tcp/localhost:7447"
    topic_frame:str = "litter/frame"
    topic_mask_binary:str = "litter/mask/binary"
    topic_mask_probabilities:str = "litter/mask/probabilities"
    topic_visualization:str = "litter/visualization"

    # OpenTelemetry setup

    SERVICE_NAME:str = "litter-detector"
    OTEL_ENDPOINT:str = "http://127.0.0.1:4317"
