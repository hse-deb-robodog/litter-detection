# Your task

> Your mission, should you choose to accept it...

## Aim - Agentic Robot System 

We enhance the remote controlled litter detection robot. The robot now should be controlled by an agentic system and interact with a human companion.

- Build a (Multi-) Agent System that controls the robot and interact with a human controller.
- The robot should be able to search litter in a predefined square space like an open field. 
- Detected litter should be marked on a map and reported to a human. The human collects the litter, later.

Reminder:

- Document the process and usage of AI during the lab task

Assumptions/Given Functions:

- The robot has a self localisation and reports its position and orientation.
- Consider the robot as blackbox, that can be controlled via messages and reports sensor information and its status via messages.
- Assume the starting point of the robot is one corner of the open field. Within this field there are no possible collisions.

```
  OPEN FIELD
  +------------------------------------------------------------------+
  |                                                                  |
  |   SEARCH ZONE                                                    |
  |   +--------------------------------------------------------+     |
  |   | [R]---->---->---->---->---->---->---->---->---->----v  |     |
  |   |  |   *              *                      *        |  |     |
  |   |  ^<----<----<----<----<----<----<----<----<----<----+  |     |
  |   |  |           *                    *                    |     |
  |   |  +---->---->---->---->---->---->---->---->---->----v   |     |
  |   |      *                  *                   *    |     |     |
  |   |  ^<----<----<----<----<----<----<----<----<----<-+     |     |
  |   |  |                *                                    |     |
  |   |  +---->---->---->---->---->---->---->---->---->-[END]  |     |
  |   +--------------------------------------------------------+     |
  |                                                                  |
  +------------------------------------------------------------------+

  [R]  = Robot Start        * = Detected Litter
  ----> = Search Path       Boustrophedon (lawnmower) pattern
```

### Work Packages

1. Layout the agents you need and define the task they need to do. 
2. Decide about the connection between robot and agents.
3. Think about the concurrent task that might be active, the movement pattern, how to start and stop the robot, interaction with a human.
4. Research solutions to achieve your functionality (e.g. Speech processing)
5. Integrate the Robot to your agentic system.

Questions you might ask yourself:

- Which component should plan the scenario?
- Which component observes the current execution?
- How is a plan represented?

## Deliverable

1. AI Usage: How did you use AI during this task? (Prompts, Agents, Pipelines, Tools, ...)
2. Illustration of the agentic architecture. 
3. Demo litter search with agentic interaction.

## Kickstart idea for a first simple system

A first simple system might look like this:

1. After the start the robot just turns on the spot and captures images.
2. In case it detects litter the robot stops and sits down.
3. After some seconds the robot stands up and continues with 1

## Guardrails

1. **Emergency stop overrides all sources** — Any agent or component receiving an `EmergencyStopCommand(stop)` must immediately cease issuing `MovementCommand`s and must not resume until `EmergencyStopCommand(release)` is received.

2. **Source authority hierarchy** — `controller` commands always override `autonomous` and `planner`. When a `MovementCommand` with `source=controller` is received, the agentic system must pause its own commands until the controller goes idle (sends a zero command or stops publishing).

3. **Speed limits for autonomous sources** — `MovementCommand`s issued with `source=autonomous` or `source=planner` must cap velocity: `|x| ≤ 0.3 m/s`, `|y| ≤ 0.3 m/s`, `|z_deg| ≤ 30 deg/s`. Human controller has no such cap.

4. **Command staleness rejection** — Discard any `MovementCommand` whose `timestamp` is older than 1 second to avoid acting on stale planner output.

5. **Action-movement interlock** — `ActionCommand`s that change robot posture (`lie_down`, `sit_down`, `stretch`) must only be sent when the last `MovementCommand` was a zero command (`is_zero() == True`). Standing back up (`stand_up`, `balance_stand`) is always allowed.

6. **Image freshness for autonomous decisions** — The litter detection agent must not issue a `MovementCommand` or mark a litter position based on an `ImageBase64` whose `timestamp` is older than 2 seconds.

7. **Search zone boundary enforcement** — The planner must track the robot's position and must not issue a `MovementCommand` that would move it outside the predefined search zone boundary. If the robot is at a boundary, only commands moving it inward are allowed.

## Example: Receiving a Camera Image via Zenoh

```python
import zenoh
from interfaces.image import ImageBase64
import numpy as np

session = zenoh.open(zenoh.Config())

def on_image(sample):
    image = ImageBase64.model_validate_json(sample.payload.to_bytes())
    # Reconstruct numpy array from raw bytes
    array = np.frombuffer(image.data, dtype=image.dtype).reshape(image.shape)
    print(f"Received image at {image.timestamp}: shape={array.shape}")
    # Pass array to litter detection model here

subscriber = session.declare_subscriber("robot/sensor/image", on_image)
```

## Example: Sending a Robot Command via Zenoh

```python
import zenoh
from interfaces.motion import MovementCommand, MovementSource, ActionCommand, ActionType, EmergencyStopCommand, EmergencyStop

session = zenoh.open(zenoh.Config())

# Move forward autonomously
cmd = MovementCommand(x=0.2, y=0.0, z_deg=0.0, source=MovementSource.planner)
session.put("robot/cmd/movement", cmd.model_dump_json())

# Rotate on the spot to scan for litter
scan = MovementCommand(x=0.0, y=0.0, z_deg=20.0, source=MovementSource.autonomous)
session.put("robot/cmd/movement", scan.model_dump_json())

# Sit down after detecting litter (only valid when is_zero())
stop = MovementCommand(source=MovementSource.autonomous)  # all zero
session.put("robot/cmd/movement", stop.model_dump_json())
action = ActionCommand(action=ActionType.sit_down)
session.put("robot/cmd/action", action.model_dump_json())

# Emergency stop
estop = EmergencyStopCommand(command=EmergencyStop.stop)
session.put("robot/cmd/estop", estop.model_dump_json())
```

## Maybe useful

- FastAPI Server Sent Events to publish updates from an API: https://fastapi.tiangolo.com/tutorial/server-sent-events/
- Harness for pydantic ai: https://github.com/pydantic/pydantic-ai-harness
- [Agentic LLM-based robotic systems for real-world applications: a review on their agenticness and ethics](https://pdfs.semanticscholar.org/e126/28d38ffb0c5e290e6519a53c4b25d128c903.pdf?_gl=1*z4di2t*_gcl_au*MTkzNjM3Mzc3NS4xNzcyODA2MTIw*_ga*MTg0NDYyNDAwNy4xNzcyODA2MTIw*_ga_H7P4ZT52H5*czE3NzY0MjcxNTUkbzckZzAkdDE3NzY0MjcxNTUkajYwJGwwJGgw)
- [Towards Embodied Agentic AI: Review and Classification of LLM- and VLM-Driven Robot Autonomy and Interaction](https://www.semanticscholar.org/reader/f571b998a35c0d2d04a41d4fb62feb77ca94fbd5)
- [Distributed AI Agents for Cognitive Underwater Robot Autonomy](https://www.semanticscholar.org/reader/120b93daa1a5a7d8324f9d6eb6374177db0402ee)
