import json
import logging
import os
import random
import sys
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import zenoh
import torch
from opentelemetry import metrics, trace
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTO_RESEARCH_DIR = PROJECT_ROOT / "auto-research"
sys.path.insert(0, str(AUTO_RESEARCH_DIR))
from train import *

# ── Inference setup ──────────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")
MODEL_NAME = os.getenv("MODEL_NAME", "models/checkpoints/best_efficientnetb4.pth")
print(f"Using model: {MODEL_NAME}")

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("detector")

# ── OpenTelemetry setup ──────────────────────────────────────────────────────
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "litter-detector")

resource = Resource.create({ResourceAttributes.SERVICE_NAME: SERVICE_NAME})

# Traces
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True))
)

trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(SERVICE_NAME)

# Metrics
metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint=OTEL_ENDPOINT, insecure=True), export_interval_millis=5000
)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter(SERVICE_NAME)

# Logs
log_provider = LoggerProvider(resource=resource)
log_provider.add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter(endpoint=OTEL_ENDPOINT, insecure=True))
)
set_logger_provider(log_provider)
logging.getLogger().addHandler(LoggingHandler(logger_provider=log_provider))

# ── Metrics instruments ───────────────────────────────────────────────────────
inference_latency = meter.create_histogram(
    "inference_duration_seconds", description="YOLO inference latency", unit="s"
)

detection_latency = meter.create_histogram(
    "detection_duration_seconds", description="Total detection latency (preprocessing + inference)", unit="s"
)

detection_counter = meter.create_counter(
    "detections_total", description="Total detections per class"
)
confidence_hist = meter.create_histogram(
    "detection_confidence", description="YOLO confidence scores", unit="1"
)
corrupt_frames = meter.create_counter(
    "corrupt_frames_total", description="Frames rejected by preprocessing"
)
frame_brightness = meter.create_histogram(
    "frame_brightness", description="Mean pixel brightness", unit="1"
)
frames_processed = meter.create_counter(
    "frames_processed_total", description="Total frames processed"
)

# ── Model ────────────────────────────────────────────────────────────────────


def load_model():
    global model, device
    logger.info(f"Loading {MODEL_NAME} model…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(MODEL_NAME)
    if checkpoint.suffix != ".pth":
        checkpoint = checkpoint.with_suffix(".pth")
    if not checkpoint.is_absolute():
        if checkpoint.parts[:2] != ("models", "checkpoints"):
            checkpoint = Path("models") / "checkpoints" / checkpoint.name
        checkpoint = PROJECT_ROOT / checkpoint
    CHECKPOINT = checkpoint
    MODEL_CLASS = EfficientNetB4UNet

    if CHECKPOINT.exists():
        model = MODEL_CLASS(dropout=DROPOUT).to(device)
        model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
        model.eval()
        n = sum(par.numel() for par in model.parameters())
        print(f"Loaded {CHECKPOINT} ({MODEL_CLASS.__name__}) on {device}  ({n:,} params)")
    else:
        print(f"{CHECKPOINT} not found")
 

# ── Frame sources ─────────────────────────────────────────────────────────────
LITTER_CLASSES = ["bottle", "can", "paper", "plastic bag", "cigarette", "cup"]

frame_queue = deque(maxlen=20)
frame_queue_lock = threading.Lock()
frame_available = threading.Event()


def on_frame(sample: zenoh.Sample) -> None:
    try:
        frame_bytes = bytes(sample.payload)
        with frame_queue_lock:
            frame_queue.append(frame_bytes)
            frame_available.set()
    except Exception as exc:
        logger.exception("Failed to enqueue frame from Zenoh")


def synthetic_frames():
    """Yield synthetic frame payloads (noise images) for demo without webcam."""
    logger.info("Running in synthetic camera mode")
    frame_id = 0
    while True:
        h, w = 480, 640
        img = np.random.randint(30, 200, (h, w, 3), dtype=np.uint8)
        success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not success:
            logger.warning("Synthetic frame encoding failed, retrying…")
            continue
        yield buf.tobytes()
        frame_id += 1
        time.sleep(0.1)  # 10 fps

def synthetic_frames():
    """Yield synthetic frame payloads (noise images) for demo without webcam."""
    logger.info("Running in synthetic camera mode")
    frame_id = 0
    while True:
        # Generate a plausible-looking noisy BGR frame
        h, w = 480, 640
        img = np.random.randint(30, 200, (h, w, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img, [zenoh.Encoding.IMAGE_JPEG, 70])
        yield buf.tobytes()
        frame_id += 1
        time.sleep(0.1)  # 10 fps


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_frame(image_bytes: bytes) -> np.ndarray | None:
    with tracer.start_as_current_span("preprocess-frame") as span:
        try:
            arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                corrupt_frames.add(1, {"reason": "decode_error"})
                span.set_status(trace.StatusCode.ERROR, "Failed to decode frame")
                return None
            brightness = float(img.mean()) / 255.0
            frame_brightness.record(brightness)
            span.set_attribute("frame.size_bytes", len(image_bytes))
            span.set_attribute("frame.brightness", round(brightness, 3))

            return img
        except Exception as exc:
            corrupt_frames.add(1, {"reason": "exception"})
            span.record_exception(exc)
            span.set_status(trace.StatusCode.ERROR, str(exc))
            raise


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(img: np.ndarray, frame_bytes: bytes) -> dict:
    with tracer.start_as_current_span("efficientnet-inference") as span:
        span.set_attribute("frame.size_bytes", len(frame_bytes))
        span.set_attribute("model.name", MODEL_NAME)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.perf_counter()
        
        # Preprocessing: Normalisieren und zu Tensor konvertieren
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
        
        duration = time.perf_counter() - t0
        
        # Verarbeite Ausgaben
        if isinstance(outputs, torch.Tensor):
            probs = torch.softmax(outputs, dim=1)
            conf_scores, class_ids = torch.max(probs, dim=1)
            conf_scores = conf_scores.cpu().numpy().tolist()
            class_ids = class_ids.cpu().numpy().tolist()
        else:
            conf_scores = []
            class_ids = []
        
        span.set_attribute("detections.count", len(conf_scores))
        span.set_attribute("inference.duration_ms", round(duration * 1000, 1))
        
        inference_latency.record(duration, {"model": MODEL_NAME})
        frames_processed.add(1)
        
        detections = []
        class_names = ["bottle", "can", "paper", "plastic bag", "cigarette", "cup"]
        
        for conf, cls_id in zip(conf_scores, class_ids):
            if conf > 0.25:  # Confidence threshold
                cls_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else "unknown"
                detection_counter.add(1, {"class": cls_name})
                confidence_hist.record(conf, {"model": MODEL_NAME})
                detections.append({"class": cls_name, "confidence": round(conf, 3)})
        
        return {
            "detections": detections,
            "latency_ms": round(duration * 1000, 1),
            "model": MODEL_NAME,
        }

# ── Synthetic inference (no YOLO) ─────────────────────────────────────────────
def run_synthetic_inference() -> dict:
    """Produce plausible detection results without a real model."""
    with tracer.start_as_current_span("yolo-inference") as span:
        t0 = time.perf_counter()
        time.sleep(random.gauss(0.04, 0.005))  # ~40ms ± 5ms
        duration = time.perf_counter() - t0

        n = random.choices([0, 1, 2, 3], weights=[0.3, 0.4, 0.2, 0.1])[0]
        detections = []
        for _ in range(n):
            cls = random.choice(LITTER_CLASSES)
            conf = max(0.25, min(0.99, random.gauss(0.72, 0.12)))
            detection_counter.add(1, {"class": cls})
            confidence_hist.record(conf, {"model": "synthetic"})
            detections.append({"class": cls, "confidence": round(conf, 3)})

        span.set_attribute("detections.count", n)
        span.set_attribute("inference.duration_ms", round(duration * 1000, 1))
        span.set_attribute("model.name", "synthetic")
        inference_latency.record(duration, {"model": "synthetic"})
        frames_processed.add(1)
        return {"detections": detections, "latency_ms": round(duration * 1000, 1), "model": "synthetic"}

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    camera_mode = os.getenv("CAMERA_MODE", "webcam").lower()
    use_real_model = camera_mode == "webcam"

    logger.info(f"Starting detector with CAMERA_MODE={camera_mode}")

    if use_real_model:
        load_model()

    zenoh_router = os.getenv("ZENOH_ROUTER", "localhost:7447")
    logger.info(f"Connecting to Zenoh router at {zenoh_router}…")

    conf = zenoh.Config()
    conf.insert_json5("connect/endpoints", json.dumps([f"tcp/{zenoh_router}"]))
    session = zenoh.open(conf)
    logger.info("Zenoh session open. Starting detection loop.")

    subscriber = None
    if use_real_model:
        subscriber = session.declare_subscriber("litter/frame", on_frame)
        logger.info("Subscribed to litter/frame")

    try:
        while True:
            if use_real_model:
                if not frame_available.wait(1.0):
                    continue
                with frame_queue_lock:
                    if not frame_queue:
                        frame_available.clear()
                        continue
                    frame_bytes = frame_queue.popleft()
                    if not frame_queue:
                        frame_available.clear()
            else:
                frame_bytes = None

            with tracer.start_as_current_span("process-frame") as root_span:
                try:
                    overall_t0 = time.perf_counter()

                    if use_real_model:
                        img = preprocess_frame(frame_bytes)
                        if img is None:
                            continue
                        result = run_inference(img, frame_bytes)
                    else:
                        result = run_synthetic_inference()

                    n = len(result["detections"])
                    logger.info(
                        f"Processed frame: {n} detection(s) in {result['latency_ms']:.1f} ms"
                    )
                    root_span.set_attribute("detections.count", n)

                    session.put(
                        "litter/detections",
                        json.dumps(result).encode(),
                    )

                    overall_latency = time.perf_counter() - overall_t0
                    detection_latency.record(overall_latency)
                    root_span.set_attribute("detection.duration_ms", round(overall_latency * 1000, 1))
                    logger.info(f"Total detection latency (preprocessing + inference): {overall_latency:.3f} s")

                except Exception as exc:
                    logger.exception("Frame processing failed")
                    root_span.record_exception(exc)
                    root_span.set_status(trace.StatusCode.ERROR, str(exc))

    except KeyboardInterrupt:
        logger.info("Beende Detector...")
    finally:
        if subscriber is not None:
            subscriber.undeclare()
        session.close()


if __name__ == "__main__":
    main()
