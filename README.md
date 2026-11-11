# ERIBench: A Comprehensive Benchmark for Embodied Reference Interaction in Human-Robot Scenarios

* **ERIBench:** the first comprehensive benchmark for Embodied Reference Interaction (ERI). It systematically evaluates the performance of MLLMs as ‘embodied brains’ across five key dimensions, covering 10 distinct tasks and 6,535 curated question-answer pairs.

## Paper

Read the paper: [ERIBench: A Comprehensive Benchmark for Embodied Reference Interaction in Human-Robot Scenarios](./Arxiv_ERIBench.pdf).

## Dataset

The full dataset is hosted on [huggingface](https://huggingface.co/datasets/ghy1/ERIBench)

<!-- ### Structure

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
``` -->

## Quick Start

### Download from Hugging Face

<!-- ```python
from datasets import load_dataset

dataset = load_dataset("ghy1/ERIBench")
``` -->

<!-- export HF_ENDPOINT=https://hf-mirror.com -->
```bash
hf download --repo-type dataset ghy1/ERIBench --local-dir ./dataset
```

### Evaluation

```bash
# Example evaluate Qwen3-VL-2B-Instruct
# inference
python inference.py \
    --model_type qwenvl \
    --model_name Qwen/Qwen3-VL-2B-Instruct \
    --testset_dir ./dataset/test \
    --image_dir ./dataset/test/images  \
    --output_dir ./predictions/Qwen3-VL-2B-Instruct \
    --temperature 0.0 \
    --max_tokens 4096 \
    --batch_size 8

# evaluate metrics
API_KEY=""
python evaluate_metrics.py \
    --predictions_path ./predictions/Qwen3-VL-2B-Instruct/ \
    --output_dir ./results/Qwen3-VL-2B-Instruct/ \
    --model_name Qwen/Qwen3-VL-2B-Instruct \
    --image_dir ./dataset/test/images \
    --api_key $API_KEY \
    --base_url "https://ark.cn-beijing.volces.com/api/v3" \
    --score_model doubao-seed-1-8-251228
```

## License

This dataset is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Citation

If this work is helpful, please kindly cite as:

```bash
@misc{gu2026eribench,
  title={{ERIBench}: A Comprehensive Benchmark for Embodied Reference Interaction in Human-Robot Scenarios},
  author={Gu, Huayu and Xia, Menghan and Huang, Shaoli and Xuan, Wenjie and Liu, Juhua and Du, Bo},
  year={2026},
  howpublished= {GitHub repository},
  url= {https://github.com/HuayuGu/ERIBench}
}
```

