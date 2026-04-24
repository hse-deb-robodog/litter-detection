import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
AUTO_RESEARCH_DIR = PROJECT_ROOT / "auto-research"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(AUTO_RESEARCH_DIR))

from config import Settings
from train import *

import zenoh
import torch
import numpy as np
from collections import deque
import threading
import time
import cv2
import json
import logging

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource

# Opentelemetry setup
settings = Settings()

resource = Resource.create({"service.name": settings.SERVICE_NAME})
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.OTEL_ENDPOINT, insecure=True)))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(settings.SERVICE_NAME)

metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=settings.OTEL_ENDPOINT, insecure=True),export_interval_millis=5000)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter(settings.SERVICE_NAME)

# Metrics

inference_latency = meter.create_histogram("inference_duration_seconds", description="Model inference latency", unit="s")
preprocessing_duration = meter.create_histogram("preprocessing_duration_seconds", description="Frame preprocessing time", unit="s")
visualization_duration = meter.create_histogram("visualization_duration_seconds", description="Mask visualization time", unit="s")
mask_resizing_duration = meter.create_histogram("mask_resizing_duration_seconds", description="Mask resizing time", unit="s")
pipeline_duration = meter.create_histogram("pipeline_duration_seconds", description="Total end-to-end pipeline duration", unit="s")
zenoh_publish_duration = meter.create_histogram("zenoh_publish_duration_seconds", description="Zenoh publication time", unit="s")
frame_queue_depth = meter.create_observable_gauge("frame_queue_depth", description="Number of frames in processing queue", unit="1", callbacks=[lambda options: [options.Observation(len(frame_queue))]])
confidence_hist = meter.create_histogram("detection_confidence", description="Detection confidence scores", unit="1")
frames_processed = meter.create_counter("frames_processed_total", description="Total frames processed")
frames_skipped = meter.create_counter("frames_skipped_total", description="Frames skipped due to age")
frames_received = meter.create_counter("frames_received_total", description="Total frames received from Zenoh")
mask_output_size = meter.create_histogram("mask_output_size_bytes", description="Size of generated mask output", unit="By")
model = None
device = None

# Performance optimization settings
FRAME_MAX_AGE_SECONDS = 0.5  # Aggressive: Skip frames older than 0.5 seconds
PROCESSING_TIMEOUT_SECONDS = 2.0  # Abort if processing takes longer than 2 seconds
INPUT_IMAGE_SIZE = 512  # Reduce resolution for faster processing
JPEG_QUALITY = 50  # Lower JPEG quality for faster encoding

def load_model():
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    logging.info(f"Loading model: {settings.MODEL_NAME} on device: {device}")
    
    # Resolve checkpoint path relative to the project root
    CHECKPOINT = PROJECT_ROOT / settings.MODEL_NAME
    
    if CHECKPOINT.exists():
        model = settings.MODEL_CLASS(dropout=settings.DROPOUT).to(device)
        model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
        model.eval()
        n = sum(par.numel() for par in model.parameters())
        logging.info(f"Loaded {CHECKPOINT} ({settings.MODEL_CLASS.__name__}) on {device}  ({n:,} params)")
    else:
        logging.error(f"{CHECKPOINT} not found")
        raise FileNotFoundError(f"Model checkpoint not found at {CHECKPOINT}")

frame_queue = deque(maxlen=20)  # Stores tuples of (frame_bytes, timestamp, frame_height, frame_width)
frame_queue_lock = threading.Lock()
frame_available = threading.Event()

def on_frame_received(sample: zenoh.Sample):
    try:
        frame_bytes = bytes(sample.payload)
        frame_timestamp = time.perf_counter()  # Capture frame arrival time
        
        # Decode frame to get original dimensions
        frame_bgr = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        frame_height, frame_width = frame_bgr.shape[:2]
        
        with tracer.start_as_current_span("receive_frame"):
            frames_received.add(1)
            with frame_queue_lock:
                frame_queue.append((frame_bytes, frame_timestamp, frame_height, frame_width))
                frame_available.set()
    except Exception as e:
        logging.exception(f"Failed to enqueue frame from Zenoh: {e}")

def preprocess_frame(frame_bytes):
    preprocess_start = time.perf_counter()
    
    img_rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    
    # Reduce resolution for faster processing
    if img_rgb.shape[0] > INPUT_IMAGE_SIZE or img_rgb.shape[1] > INPUT_IMAGE_SIZE:
        scale = min(INPUT_IMAGE_SIZE / img_rgb.shape[0], INPUT_IMAGE_SIZE / img_rgb.shape[1])
        new_size = (int(img_rgb.shape[1] * scale), int(img_rgb.shape[0] * scale))
        img_rgb = cv2.resize(img_rgb, new_size, interpolation=cv2.INTER_LINEAR)
    
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    preprocess_duration = time.perf_counter() - preprocess_start
    preprocessing_duration.record(preprocess_duration)
    
    return img_tensor.to(device)

def inference(frame_bytes: bytes):
   
        mask_probabilities = None
        mask_binary = None
        
        with tracer.start_as_current_span("inference") as span:
            span.set_attribute("frame_size_bytes", len(frame_bytes))
            span.set_attribute("model_name", settings.MODEL_NAME)
            start_time = time.perf_counter()

            
            try:

                img_tensor = preprocess_frame(frame_bytes)
                with torch.no_grad():
                    output = model(img_tensor)
                
                duration = time.perf_counter() - start_time

                if isinstance(output, torch.Tensor):
                    # Für binäre Segmentierung: Sigmoid anwenden, um Wahrscheinlichkeiten zu bekommen
                    mask_probabilities = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()
                    # Optional: Threshold anwenden, um binäre Maske zu bekommen (z.B. > 0.5)
                    mask_binary = (mask_probabilities > settings.THRESHOLD).astype(np.uint8)
                else:
                    mask_probabilities = None
                    mask_binary = None
                
                span.set_attribute("inference_duration_seconds", round(duration*1000, 1))
                span.set_attribute("mask_shape", str(mask_probabilities.shape if mask_probabilities is not None else "None"))
                inference_latency.record(duration, {"model_name": settings.MODEL_NAME})
                frames_processed.add(1)
                
                logging.info(f"Maske erhalten: Shape {mask_probabilities.shape}, Binary Mask Shape {mask_binary.shape}")

            except Exception as e:
                logging.exception(f"Error during inference: {e}")
                span.record_exception(e)
            
            return mask_probabilities, mask_binary
            

def visualize_mask(frame_bytes: bytes, mask_binary: np.ndarray, original_frame_height: int, original_frame_width: int, alpha: float = 0.5):
    viz_start = time.perf_counter()
    try:
        # Dekodiere Frame zu OpenCV-Format (BGR)
        frame_bgr = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize Maske auf Original-Frame-Größe
        mask_height, mask_width = mask_binary.shape[:2]
        
        if (mask_height, mask_width) != (original_frame_height, original_frame_width):
            resize_start = time.perf_counter()
            mask_binary = cv2.resize(mask_binary, (original_frame_width, original_frame_height), interpolation=cv2.INTER_NEAREST)
            mask_resizing_duration.record(time.perf_counter() - resize_start)
        
        # Erstelle rotes Overlay für erkannte Litter-Bereiche
        overlay = frame_rgb.copy()
        # Rot im RGB-Format
        overlay[mask_binary == 1] = [255, 0, 0]
        
        # Blende Overlay mit Original zusammen
        visualized = cv2.addWeighted(frame_rgb, 1 - alpha, overlay, alpha, 0)
        
        logging.debug(f"Visualisierung erstellt: {visualized.shape}")
        visualization_duration.record(time.perf_counter() - viz_start)
        return visualized
        
    except Exception as e:
        logging.exception(f"Error during mask visualization: {e}")
        return None

def main():
    load_model()

    conf = zenoh.Config()
    conf.insert_json5("connect/endpoints", json.dumps([settings.ZENOH_ROUTER]))
    z = zenoh.open(conf)
    frame_sub = z.declare_subscriber(settings.topic_frame, on_frame_received)
    logging.info(f"Subscribed to Zenoh topic: {settings.topic_frame}")
    
    frame_cache = None
    last_send_time = 0
    min_interval = 0.02  # Limit sends to ~50 FPS max
    
    try:
        while True:
            if not frame_available.wait(timeout=1):
                continue
            
            frame_bytes = None
            frame_timestamp = None
            frame_height = None
            frame_width = None
            frames_to_skip = 0
            
            # Extract the most recent frame, skip old ones
            with frame_queue_lock:
                while frame_queue:
                    frame_bytes, frame_timestamp, frame_height, frame_width = frame_queue.popleft()
                    current_time = time.perf_counter()
                    frame_age = current_time - frame_timestamp
                    
                    # Aggressively skip old frames
                    if frame_age > FRAME_MAX_AGE_SECONDS:
                        frames_to_skip += 1
                        if logging.root.level <= logging.DEBUG:
                            logging.debug(f"Skipping frame (age: {frame_age:.3f}s > {FRAME_MAX_AGE_SECONDS}s)")
                        frames_skipped.add(1)
                        frame_bytes = None
                        continue
                    else:
                        break
                
                if not frame_queue:
                    frame_available.clear()
            
            # Skip processing if we couldn't find a fresh frame
            if frame_bytes is None:
                continue
            
            # Rate limiting to avoid overwhelming the visualizer
            current_time = time.perf_counter()
            if current_time - last_send_time < min_interval:
                continue
            last_send_time = current_time
            
            with tracer.start_as_current_span("process_frame") as root_span:
                overall_start = time.perf_counter()

                mask_probabilities, mask_binary = inference(frame_bytes)
                
                # Resize masks to original frame dimensions BEFORE sending via Zenoh
                # This ensures the visualizer receives masks in the correct size
                resized_mask_probs = None
                resized_mask_binary = None
                
                if mask_probabilities is not None:
                    mask_h, mask_w = mask_probabilities.shape[:2]
                    logging.info(f"Probability mask shape: {mask_probabilities.shape}, target: ({frame_height}, {frame_width})")
                    if (mask_h, mask_w) != (frame_height, frame_width):
                        # Use cv2.resize with correct order: (width, height) for resize, but numpy shape is (height, width)
                        resize_start = time.perf_counter()
                        resized_mask_probs = cv2.resize(mask_probabilities, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
                        mask_resizing_duration.record(time.perf_counter() - resize_start)
                        logging.info(f"Resized probability mask to: {resized_mask_probs.shape}")
                    else:
                        resized_mask_probs = mask_probabilities
                
                if mask_binary is not None:
                    mask_h, mask_w = mask_binary.shape[:2]
                    logging.info(f"Binary mask shape: {mask_binary.shape}, target: ({frame_height}, {frame_width})")
                    if (mask_h, mask_w) != (frame_height, frame_width):
                        # Use cv2.resize with correct order: (width, height) for resize, but numpy shape is (height, width)
                        resize_start = time.perf_counter()
                        resized_mask_binary = cv2.resize(mask_binary, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                        mask_resizing_duration.record(time.perf_counter() - resize_start)
                        logging.info(f"Resized binary mask to: {resized_mask_binary.shape}")
                    else:
                        resized_mask_binary = mask_binary
                
                # Visualisiere die Maske mit korrekter Original-Größe
                visualized_img = visualize_mask(frame_bytes, resized_mask_binary, frame_height, frame_width, alpha=0.5) if resized_mask_binary is not None else None
                
                # Sende Ergebnisse über Zenoh (use resized versions!)
                publish_start = time.perf_counter()
                if resized_mask_probs is not None:
                    z.put(settings.topic_mask_probabilities, resized_mask_probs.tobytes())
                    mask_output_size.record(resized_mask_probs.nbytes, {"output_type": "mask_probabilities"})
                    logging.info(f"Sent probability mask (bytes: {resized_mask_probs.nbytes}, shape: {resized_mask_probs.shape})")
                
                if resized_mask_binary is not None:
                    z.put(settings.topic_mask_binary, resized_mask_binary.tobytes())
                    mask_output_size.record(resized_mask_binary.nbytes, {"output_type": "mask_binary"})
                    logging.info(f"Sent binary mask (bytes: {resized_mask_binary.nbytes}, shape: {resized_mask_binary.shape})")
                
                if visualized_img is not None:
                    # Konvertiere RGB zu BGR für JPEG Encoding
                    visualized_bgr = cv2.cvtColor(visualized_img, cv2.COLOR_RGB2BGR)
                    success, buf = cv2.imencode(".jpg", visualized_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    z.put(settings.topic_visualization, buf.tobytes(), encoding=zenoh.Encoding.IMAGE_JPEG)
                    mask_output_size.record(buf.nbytes, {"output_type": "visualization"})
                    logging.debug(f"Sent visualization: {visualized_bgr.shape}")
                
                publish_duration = time.perf_counter() - publish_start
                zenoh_publish_duration.record(publish_duration)
                
                overall_duration = time.perf_counter() - overall_start
                pipeline_duration.record(overall_duration)
                root_span.set_attribute("overall_processing_duration_seconds", round(overall_duration*1000, 1))
                root_span.set_attribute("zenoh_publish_duration_seconds", round(publish_duration*1000, 1))
                root_span.set_attribute("frame_height", frame_height)
                root_span.set_attribute("frame_width", frame_width)
                
                # Warn if processing takes too long
                if overall_duration > PROCESSING_TIMEOUT_SECONDS:
                    logging.warning(f"Processing took {overall_duration:.3f}s (> {PROCESSING_TIMEOUT_SECONDS}s threshold)")
                    frames_skipped.add(5)  # Penalize slow processing
                
                if frames_to_skip > 0:
                    logging.info(f"Frame verarbeitet ({overall_duration:.3f}s) | {frames_to_skip} übersprungen")
                else:
                    logging.info(f"Frame verarbeitet ({overall_duration:.3f}s)")


       
    except KeyboardInterrupt:
        logging.info("Shutting down...")

if __name__ == "__main__":
    main()
