# Ordnerstruktur
## Prompt:

Du bist ein erfahrener Software-Architekt mit Spezialisierung auf Repository-Strukturierung und Codebase-Organisation. Ich möchte, dass du mein Repository refaktorierst und eine saubere, übersichtliche Ordnerstruktur erstellst.

**Aktuelles Problem:** Bilder, Dokumente, trainierte Modelle und Skripte liegen unstrukturiert im selben Ordner.

**Konkrete Aufgaben:**

1. **Dokumentation verschieben:** Verschiebe alle Dokumentationsdateien in einen eigenen `/docs`-Ordner. Aktualisiere dabei alle internen Verweise und Links in den `.md`-Dateien, sodass keine toten Links entstehen.

2. **Auto-Research-Ordner erstellen:** Verschiebe die Dateien `prepare.py`, `train.py` und `program.md` in einen neuen Ordner `/auto-research`. Diese Dateien gehören zum [AutoResearch-Workflow](https://github.com/karpathy/autoresearch) und sollen logisch gruppiert werden.

3. **Vorschläge für verbleibende Dateien:** Analysiere die übrigen Dateien (Bilder, trainierte Modelle, sonstige Skripte) und schlage eine sinnvolle Ordnerstruktur vor — z. B. `/assets`, `/models`, `/scripts` oder ähnliches, begründet nach gängigen Best Practices.

**Für jede Änderung liefere:**
- Den genauen `mv`-Befehl oder die Schritt-für-Schritt-Anweisung zum Verschieben
- Aktualisierte Dateipfade für alle betroffenen internen Links in Markdown-Dateien
- Eine Übersicht der finalen Ordnerstruktur als ASCII-Baum

Zeige mir zum Abschluss die vollständige neue Verzeichnisstruktur des Repositorys.

## Auswertung

[Ausführliche Beschreibung der Metric](prompt_metricen.md)

| Metric                              | Score         |
|-------------------------------------|---------------|
| **Tool used**                       | Codex ChatGPT |
| **Error Rate (0 - 4)**              | 4             |
| **Code Quality (0 - 4)**            | 4             |
| **Discrepancy from Prompt (0 - 4)** | 4             |
| **Notes**                           | -             |
