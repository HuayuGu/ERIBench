# ERIBench: Embodied Robot Interaction Benchmark

ERIBench is a multimodal benchmark for evaluating vision-language models in embodied human-robot interaction scenarios. Each sample presents a first-person robot perspective image where a human user points at a target object, paired with task instructions that test various capabilities.

## Overview

Modern vision-language models (VLMs) are increasingly deployed in embodied robotics settings. ERIBench provides a comprehensive evaluation suite covering 10 distinct tasks that assess a model's ability to understand human-robot interaction scenes, including safety awareness, spatial reasoning, visual grounding, and task planning.

## Dataset

The full dataset is hosted on Hugging Face: [ghy1/ERIBench](https://huggingface.co/datasets/ghy1/ERIBench)

### Structure

```
ERIBench/
├── train/
│   ├── images/            
│   ├── safety_boundary.json
│   ├── spatial_relationship.json
│   └── visual_grounding.json
└── test/
    ├── images/           
    ├── intent_inference.json
    ├── object_color.json
    ├── object_function.json
    ├── object_material.json
    ├── object_shape.json
    ├── safety_boundary.json
    ├── safety_judgment.json
    ├── spatial_relationship.json
    ├── task_planning.json
    └── visual_grounding.json
```

### Tasks

| Task | Train | Test | Description |
|------|-------|------|-------------|
| Safety Boundary | 4,556 | 573 | Determine if executing an instruction violates human-robot safety boundaries |
| Spatial Relationship | 10,811 | 943 | Reason about spatial relations between the user, objects, and the robot |
| Visual Grounding | 2,278 | 573 | Localize the pointed object with bounding box coordinates |
| Safety Judgment | — | 1,008 | Judge whether a task is safe to execute |
| Intent Inference | — | 573 | Infer the user's intent from pointing gestures and instructions |
| Object Color | — | 573 | Identify the color of the pointed object |
| Object Function | — | 573 | Identify the function of the pointed object |
| Object Material | — | 573 | Identify the material of the pointed object |
| Object Shape | — | 573 | Identify the shape of the pointed object |
| Task Planning | — | 573 | Generate step-by-step plans for task execution |

### Data Format

Each JSON file contains a list of samples:

```json
{
  "instruction": "Task instruction with context",
  "response": "Expected response",
  "image": ["image_filename.jpg"]
}
```

For visual grounding, additional fields are included:

```json
{
  "bbox": [x_min, y_min, x_max, y_max],
  "label": "object label"
}
```

## Quick Start

### Download from Hugging Face

```python
from datasets import load_dataset

dataset = load_dataset("ghy1/ERIBench")
```

### Evaluation

```python

```

## License

This dataset is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
