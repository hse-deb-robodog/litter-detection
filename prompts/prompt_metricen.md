## Metrics
### Error Rate:
- 0: Code ist nicht lauffähig (Syntaxfehler, fehlende Imports > kritische fehler)
- 1: Code startet, hat aber Runtime Fehler bei normaler Benutzung (bricht ab)
- 2: Code läuft, liefert bei mind. einem normalen Testfall ein falsches Ergebnis (Logikfehler)
- 3: Code funktioniert weitgehend korrekt, wenige spezielle Edge Cases werden nicht korrekt behandelt
- 4: Code läuft fehlerfrei, robust und besteht alle Testfälle mit Edge Cases

### Code Qualität:
- 0: Kein erkennbares Strukturkonzept, undeutliche Variablennamen, keine Kommentare
- 1: Alles in einer Funktion / Datei, inkonsistente Benennung
- 2: Grundstruktur vorhanden (Funktionen, Klassen), deutliche Stilmängel, fehlende Kommentare
- 3: Sinnvolle Aufteilung, 1-2 Stilmängel
- 4: Modularer Aufbau, klare Namensgebung, sinnvolle Kommentare, an Best Practices gehalten

### Abweichung vom Prompt:
- 0: Ergebnis komplett abweichend
- 1: Teilweise umgesetzt, vieles fehlend
- 2: Grundlage umgesetzt, größere Teile fehlen
- 3: Kleine Details fehlen
- 4: Alle Anforderungen umgesetzt
---
## Template 

### Prompt: 
...

| Metric                              | Score |
|-------------------------------------|-------|
| **Tool used**                       |       |
| **Error Rate (0 - 4)**              |       |
| **Code Quality (0 - 4)**            |       |
| **Discrepancy from Prompt (0 - 4)** |       |
| **Notes**                           |       |