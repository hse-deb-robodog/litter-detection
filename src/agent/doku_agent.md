# Agentsaufbau
Hier sollen die Überlegungen und Definitionen dokumentiert werden, die für die Aufgabe in betracht gezogen werden. Bezogen wird sich hier auf die Aufgabenstellung aus [dem Arbeitspaket für Labor 2](../../docs/student_task_2.md).

## Aufgaben:
1. [x] Legen Sie fest, welche Agenten Sie benötigen, und definieren Sie die Aufgaben, die diese ausführen sollen. 
2. [] Legen Sie die Verbindung zwischen Roboter und Agenten fest.
3. [] Überlegen Sie sich, welche parallelen Aufgaben möglicherweise aktiv sind, welche Bewegungsmuster erforderlich sind, wie der Roboter gestartet und gestoppt wird und wie die Interaktion mit einem Menschen aussehen soll.
4. [] Recherchieren Sie Lösungen, um Ihre Funktionalität zu erreichen (z. B. Sprachverarbeitung).
5. [] Integrieren Sie den Roboter in Ihr Agentensystem.

Fragen, die Sie sich stellen könnten:

- Welche Komponente sollte das Szenario planen?
- Welche Komponente beobachtet die aktuelle Ausführung?
- Wie wird ein Plan dargestellt?

## Aufgabe 1
### Erste Überlegung:
```
Agent --- Tools
            |--- **Laufen**: Geradeauslaufen von Punkt A nach B.
            |--- **Drehen**: Wenn B erreicht drehen.
            |--- **Scan**: Während des laufens die Umgebung nach Müll Scannen.
                    |------ **Save & Send**: Bei erfolgreicher erkennung, Position speichern & senden.
                    |------ **Emote**: Bewegung & Sound bei erfolgreicher erkennung durchführen.
```

### Update Agentstruktur

![Agentstruktur](../../docs/images/agentenstruktur.png)