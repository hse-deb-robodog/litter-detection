# Anpssung auf Epochen statt Zeit

## Prompt:

Sie sind ein erfahrener ML-Engineer mit fundiertem Wissen in PyTorch-Training-Pipelines und MLflow-Experiment-Tracking. Ich habe eine `train.py`-Datei für ein Müllerkennungsmodell, die ich umschreiben möchte.

**Aktuelles Problem:** Das Training wird nach einer festen Trainingszeit abgebrochen. Das ist ungeeignet, da die Trainingsgeschwindigkeit je nach Hardware, Modellgröße und weiteren Faktoren variiert – besonders relevant, weil mehrere Studierende mit unterschiedlichen Rechnern dasselbe Modell trainieren.

**Gewünschte Änderungen an `train.py`:**

1. Ersetze das zeitbasierte Abbruchkriterium durch eine **konfigurierbare Epochenanzahl**, die als Parameter übergeben wird (z. B. via Kommandozeilenargument oder Konfigurationsvariable).
2. Das Training soll **exakt nach Erreichen der angegebenen Epochenanzahl** beendet werden.
3. **MLflow-Logging** soll aktualisiert werden, sodass die Trainingsdauer in Epochen gemessen und protokolliert wird – nicht mehr in Zeit.

**Zusätzliche Frage:** Falls es eine bessere Methode gibt, um trainierte Modelle aus verschiedenen Trainingsläufen und von verschiedenen Rechnern fair miteinander zu vergleichen (z. B. über MLflow Metrics, Model Registry oder andere Tracking-Ansätze), erkläre mir diese bitte und integriere sie gegebenenfalls in die Lösung.

Schreibe die vollständige überarbeitete `train.py` und erkläre kurz die vorgenommenen Änderungen. Meine aktuelle `train.py` ist unten beigefügt: C:\Users\robin\Documents\26SS-KI_Systeme\Allgemein\Labor\litter-detection\auto-research\train.py

## Auswertung

[Ausführliche Beschreibung der Metric](prompt_metricen.md)

| Metric                              | Score         |
|-------------------------------------|---------------|
| **Tool used**                       | Codex ChatGPT |
| **Error Rate (0 - 4)**              | 4             |
| **Code Quality (0 - 4)**            | 4             |
| **Discrepancy from Prompt (0 - 4)** | 4             |
| **Notes**                           | -             |
