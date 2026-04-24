# Your task

> Your mission, should you choose to accept it...
> Deine Mission, solltest du sie annehmen...

## Aim / Ziel - Litter Detection Function and Remote Operation

Build a litter detection system based on the proposed training history. This system should use images from the robodog and detect litter in it.

- While the operator controls the robot the dog should make some noise, if it detects litter.
- The system should be better than the proposed baseline and operate in realtime on the robodog hardware.
- The system should offer a possibility to identify and investigate possible wrong litter detections.

---
Entwickeln Sie ein System zur Erkennung von Abfall auf der Grundlage des vorgeschlagenen Trainingsverlaufs. Dieses System soll Bilder des Robodogs verwenden und darin Abfall erkennen.

- Während der Bediener den Roboter steuert, soll der Hund ein Geräusch von sich geben, wenn er Abfall erkennt.
- Das System soll besser sein als die vorgeschlagene Basisversion und in Echtzeit auf der Robodog-Hardware laufen.
- Das System soll die Möglichkeit bieten, mögliche Fehlmeldungen bei der Abfallerkennung zu identifizieren und zu untersuchen.
---

Reminder:

- Document the process and usage of AI during the lab task

Assumptions:

- litter can only be on the ground
- litter has a sufficient size (amount of pixel)

---
Zur Erinnerung:

- Dokumentiere den Einsatz und die Verwendung von KI während der Laborübung

Annahmen:

- Der Abfall befindet sich ausschließlich auf dem Boden
- Der Abfall hat eine ausreichende Größe (Pixelanzahl)
---

### Work Packages / Arbeitspakete

1. Understand the provided repository and steps taken by the system. Understand the approach of automated research 
2. Identify improvement points and improve the solution (everything allowed)
3. Compare the model with the alternative approach fine-tuning a yolo-model.
4. Prepare the inference system by connecting a webcam via eclipse zenoh.
5. Add an  application monitoring stack with suitable dashboards according to the introduced stack in the lecture

Optional:

1. Tune the system by adding additional perception approaches like an open word object detector to improve the systems performance.

---
1. Machen Sie sich mit dem bereitgestellten Repository und den vom System durchgeführten Schritten vertraut. Verstehen Sie den Ansatz der automatisierten Forschung. (x)
2. Identifizieren Sie Verbesserungspotenziale und optimieren Sie die Lösung (alle Mittel sind erlaubt).
3. Bereiten Sie das Modell mithilfe von TensorRT für die Inferenz auf der Jetson-Hardware vor. (geht noch nicht)
4. Vergleichen Sie das Modell mit dem alternativen Ansatz, bei dem ein YOLO-Modell feinabgestimmt wird.
5. Bereiten Sie das Inferenzsystem vor, indem Sie eine Webcam über Eclipse Zenoh anschließen.
6. Fügen Sie einen  Anwendungsüberwachungs-Stack mit geeigneten Dashboards hinzu, entsprechend dem in der Vorlesung vorgestellten Stack.

Optional:

1. Optimieren Sie das System, indem Sie zusätzliche Erkennungsansätze wie einen Detektor für offene Wortobjekte hinzufügen, um die Leistung des Systems zu verbessern.
---

## Deliverable / Ergebnis

1. AI Usage: How did you use AI during this task? (Prompts, Agents, Pipelines, Tools, ...)
2. Identified improvement points and result, with of these points worked out and which did not.
3. Demo of the litter detection model with reported IoU
4. Webcam Demo

---
1. Einsatz von KI: Wie haben Sie KI bei dieser Aufgabe eingesetzt? (Prompts, Agenten, Pipelines, Tools, ...)
2. Ermittelte Verbesserungsmöglichkeiten und Ergebnisse: Welche dieser Punkte konnten umgesetzt werden und welche nicht?
3. Demo des Modells zur Abfallerkennung mit Angabe des IoU-Werts
4. Webcam-Demo
---

## Camera-Sensor / Kamerasensor
## Starting Questions for finding Improvements

Here are some starting questions to assess the current approach:

1. Is a time cap a good idea while comparing different encoder backbones?
2. Is the efficientNet_B4 too big for the amount of training data?
3. How good is the labeling of the data?

## Camera-Sensor

- Use either the image from the robodog camera or the image from the webcam

---
- Verwenden Sie entweder das Bild der Robodog-Kamera oder das Bild der Webcam
---

## Guardrails / Leitplanken

- track the experiments with mlflow
- Compare the U-Net based approach with the yolo based fine tuning approach

---
- Verfolgen Sie die Experimente mit mlflow
- Vergleichen Sie den U-Net-basierten Ansatz mit dem YOLO-basierten Feinabstimmungsansatz
---

## Zenoh Kickstart

We use zenoh as router. To start it as container use:

```bash
docker run --init -p 7447:7447/tcp -p 8000:8000/tcp eclipse/zenoh
```

Use [zenoh-hammer](https://github.com/sanri/zenoh-hammer) to show and debug the messages

Basic tutorial for zenoh:

- Getting started with zenoh: https://zenoh.io/docs/getting-started/first-app/
- Webcam demo: https://github.com/eclipse-zenoh/zenoh-demos/tree/main/computer-vision/zcam/zcam-python
