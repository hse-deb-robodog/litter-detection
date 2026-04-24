import json
import logging
import os
import sys
import time

import cv2
import zenoh
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import Settings

settings = Settings()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("webcam")


def publish_webcam():
    logger.info(f"Connecting to Zenoh router at {settings.ZENOH_ROUTER}…")
    conf = zenoh.Config()
    conf.insert_json5("connect/endpoints", json.dumps([settings.ZENOH_ROUTER]))
    session = zenoh.open(conf)
    logger.info("Zenoh session open. Publishing webcam frames to litter/frame.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam. Bitte überprüfe die Kamera.")
        session.close()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Webcam read failed, retrying…")
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (640, 480))
            success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                logger.warning("JPEG encoding failed, überspringe Frame.")
                continue

            session.put(settings.topic_frame, buf.tobytes())
            logger.debug(f"Published frame to {settings.topic_frame} (%d bytes)", len(buf))
            time.sleep(1 / 10)
    finally:
        cap.release()
        session.close()
        logger.info("Beende Webcam-Publisher.")


if __name__ == "__main__":
    publish_webcam()
