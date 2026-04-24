#!/usr/bin/env python3
import json
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import zenoh


@dataclass
class DetectionResult:
    detections: list = field(default_factory=list)
    latency_ms: float = 0.0
    model: str = "unknown"


class LitterVisualizer:
    CLASS_COLORS = {
        "plastic bag": (0, 165, 255),
        "bottle":      (0, 255, 0),
        "cup":         (255, 0, 255),
        "can":         (255, 255, 0),
        "paper":       (0, 255, 255),
        "cigarette":   (0, 0, 255),
        "default":     (255, 255, 255),
    }

    def __init__(self):
        self.pending_frame:      Optional[np.ndarray]     = None
        self.pending_detections: Optional[DetectionResult] = None

        # Fertig gematchtes Paar – vom Hauptthread abgeholt
        self.ready_frame:      Optional[np.ndarray]     = None
        self.ready_detections: Optional[DetectionResult] = None

        self.lock    = threading.Lock()
        self.running = True

    # ------------------------------------------------------------------ #
    #  Zenoh-Callbacks  (laufen in fremden Threads)                        #
    # ------------------------------------------------------------------ #

    def on_frame(self, sample: zenoh.Sample) -> None:
        try:
            img_array = np.frombuffer(bytes(sample.payload), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                return
            with self.lock:
                self.pending_frame = frame
                self._try_match()
        except Exception as e:
            print(f"Fehler beim Frame-Dekodieren: {e}")

    def on_detections(self, sample: zenoh.Sample) -> None:
        try:
            data   = json.loads(bytes(sample.payload).decode("utf-8"))
            result = DetectionResult(
                detections=data.get("detections", []),
                latency_ms=data.get("latency_ms", 0.0),
                model=data.get("model", "unknown"),
            )
            with self.lock:
                self.pending_detections = result
                self._try_match()
        except Exception as e:
            print(f"Fehler beim Detektions-Dekodieren: {e}")

    def _try_match(self) -> None:
        """Muss unter self.lock aufgerufen werden."""
        if self.pending_frame is not None and self.pending_detections is not None:
            # Fertiges Paar für den Hauptthread bereitstellen
            self.ready_frame      = self.pending_frame
            self.ready_detections = self.pending_detections
            self.pending_frame      = None
            self.pending_detections = None

    # ------------------------------------------------------------------ #
    #  Hauptthread                                                         #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        print("Starte Visualisierung... (ESC oder 'q' zum Beenden)")
        cv2.namedWindow("Litter Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Litter Detection", 1280, 720)

        placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Warte auf Daten...", (500, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128, 128, 128), 3)
        cv2.imshow("Litter Detection", placeholder)

        while self.running:
            # Bereitgestelltes Paar holen (kurze Lock-Zeit)
            with self.lock:
                frame      = self.ready_frame
                detections = self.ready_detections
                self.ready_frame      = None
                self.ready_detections = None

            if frame is not None and detections is not None:
                display = self._render_frame(frame, detections)
                cv2.imshow("Litter Detection", display)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                self.running = False

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    #  Rendering (unverändert)                                             #
    # ------------------------------------------------------------------ #

    def _render_frame(self, frame: np.ndarray, detections: DetectionResult) -> np.ndarray:
        display = frame.copy()
        height, width = display.shape[:2]

        cv2.rectangle(display, (0, 0), (width, 40), (0, 0, 0), -1)
        info_text = (f"Model: {detections.model} | "
                     f"Latency: {detections.latency_ms:.1f}ms | "
                     f"Detections: {len(detections.detections)}")
        cv2.putText(display, info_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y_offset = 70
        for det in detections.detections:
            class_name = det.get("class", "unknown")
            confidence = det.get("confidence", 0.0)
            color      = self.CLASS_COLORS.get(class_name, self.CLASS_COLORS["default"])

            text = f"{class_name}: {confidence:.1%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

            cv2.rectangle(display, (10, y_offset - th - 5), (20 + tw, y_offset + 5), (0, 0, 0), -1)
            cv2.rectangle(display, (10, y_offset - th - 3), (15, y_offset + 3), color, -1)
            cv2.putText(display, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            bar_w = int(150 * confidence)
            x0 = 20 + tw + 10
            cv2.rectangle(display, (x0, y_offset - 12), (x0 + bar_w, y_offset), color, -1)
            cv2.rectangle(display, (x0, y_offset - 12), (x0 + 150,   y_offset), color, 1)

            y_offset += 40

        if not detections.detections:
            cv2.putText(display, "Keine Objekte erkannt", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        return display


def main():
    config  = zenoh.Config()
    print("Verbinde mit Zenoh...")
    session = zenoh.open(config)
    print("Zenoh-Session geöffnet")

    visualizer = LitterVisualizer()

    sub_frame      = session.declare_subscriber("litter/frame",      visualizer.on_frame)
    sub_detections = session.declare_subscriber("litter/detections",  visualizer.on_detections)

    print("\n" + "=" * 50)
    print("Litter Detection Visualizer läuft")
    print("Topics: litter/frame, litter/detections")
    print("=" * 50 + "\n")

    try:
        visualizer.run()          # blockiert im Hauptthread
    except KeyboardInterrupt:
        print("\nBeende...")
    finally:
        sub_frame.undeclare()
        sub_detections.undeclare()
        session.close()
        print("Zenoh-Session geschlossen")


if __name__ == "__main__":
    main()