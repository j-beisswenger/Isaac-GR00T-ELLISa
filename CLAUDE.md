# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NVIDIA Isaac GR00T N1.5 is an open foundation model for generalist humanoid robot control. The model combines a vision-language foundation model (Eagle 2.5) with a diffusion transformer head that denoises continuous actions. It processes multimodal inputs (language, images, robot state) to generate robot actions across different embodiments.

The codebase supports:
- **Inference**: Running pre-trained models for robot control
- **Fine-tuning**: Adapting the model to new tasks, environments, or robot embodiments
- **Deployment**: TensorRT optimization for Jetson platforms (Orin, Thor)
- **Evaluation**: Testing on simulation environments and real robots

## Installation

### Standard Installation
```bash
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
```

### Platform-Specific Installation
```bash
# For Jetson Orin deployment
pip install -e .[orin]

# For Thor deployment
pip install -e .[thor]

# For deployment with TensorRT
pip install -e .[deploy]
```

**Important**: CUDA 12.4 is recommended. For CUDA 11.8, manually install `flash-attn==2.8.2`.

## Common Development Commands

### Code Quality and Testing
```bash
# Format code
make format                    # Run isort and black
isort .                       # Sort imports
black .                       # Format code

# Run checks (lint, format check, and tests)
make run-checks               # Run all checks
ruff check . --fix           # Lint with auto-fix
pytest -v --color=yes tests/ # Run all tests
pytest -v tests/test_dataset.py  # Run specific test

# Build package
make build
```

### Loading Datasets
```bash
# Load and visualize a dataset
python scripts/load_dataset.py --dataset-path ./demo_data/robot_sim.PickNPlace
```

### Running Inference

**Server mode** (loads model and waits for requests):
```bash
python scripts/inference_service.py \
    --model-path nvidia/GR00T-N1.5-3B \
    --server
```

**Client mode** (sends test observations to server):
```bash
python scripts/inference_service.py --client
```

**Evaluation** (offline evaluation on a dataset):
```bash
python scripts/eval_policy.py \
    --plot \
    --model-path nvidia/GR00T-N1.5-3B \
    --dataset-path ./demo_data/robot_sim.PickNPlace
```

### Fine-Tuning

**Basic fine-tuning**:
```bash
python scripts/gr00t_finetune.py \
    --dataset-path ./demo_data/robot_sim.PickNPlace \
    --num-gpus 1

# View all options
python scripts/gr00t_finetune.py --help
```

**Multi-dataset fine-tuning**:
```bash
python scripts/gr00t_finetune.py \
    --dataset-path <DATASET1> <DATASET2> <DATASET3> \
    --num-gpus 1
```

**LoRA fine-tuning** (for limited GPU memory):
```bash
python scripts/gr00t_finetune.py \
    --dataset-path ./demo_data/robot_sim.PickNPlace \
    --lora_rank 64 \
    --lora_alpha 128 \
    --num-gpus 1
```

**RTX 4090 fine-tuning** (avoid OOM):
```bash
python scripts/gr00t_finetune.py \
    --dataset-path ./demo_data/robot_sim.PickNPlace \
    --no-tune_diffusion_model \
    --num-gpus 1
```

### Deployment

**Export to ONNX and TensorRT**:
```bash
cd deployment_scripts

# Export model to ONNX
python export_onnx.py --model-path nvidia/GR00T-N1.5-3B

# Build TensorRT engine
bash build_engine.sh

# Run TensorRT inference
python gr00t_inference.py
```

See `deployment_scripts/README.md` for detailed deployment instructions.

## High-Level Architecture

### Three-Tier Data Flow

1. **Dataset Layer** (`gr00t/data/`):
   - `LeRobotSingleDataset`: Loads a single robot dataset in LeRobot V2 format with GR00T extensions
   - `LeRobotMixtureDataset`: Combines multiple datasets with different embodiments and sampling weights
   - `ModalityConfig`: Defines which modalities (video, state, action, language) and temporal indices to use
   - `modality.json`: Custom metadata file that maps concatenated arrays to semantic fields

2. **Transform Pipeline** (`gr00t/data/transform/`):
   - **Video transforms**: VideoToTensor → VideoCrop → VideoResize → VideoColorJitter → VideoToNumpy
   - **State/Action transforms**: StateActionToTensor → StateActionTransform (normalization)
   - **ConcatTransform**: Concatenates multiple modalities in specified order
   - **GR00TTransform**: Final transform that pads and prepares data for model input

3. **Model Layer** (`gr00t/model/`):
   - **Backbone** (`gr00t/model/backbone/`): Eagle 2.5 VLM (frozen during fine-tuning)
   - **Projector**: MLP that connects VLM output to action space
   - **Action Head** (`gr00t/model/action_head/`): Diffusion transformer for action generation
     - `FlowMatchingActionHead`: Flow matching loss and sampling
     - `CrossAttentionDiT`: Denoising transformer with cross-attention to VLM features

### Embodiment System

The model supports multiple robot embodiments through specialized action heads. Each embodiment is identified by an `EmbodimentTag` (defined in `gr00t/data/embodiment_tags.py`):

- **`EmbodimentTag.GR1`**: Fourier GR1 humanoid with dexterous hands (absolute joint control)
- **`EmbodimentTag.OXE_DROID`**: Single-arm robots with delta end-effector control
- **`EmbodimentTag.GENIE1_GRIPPER`**: Agibot Genie-1 humanoid with grippers (absolute joint control)
- **`EmbodimentTag.NEW_EMBODIMENT`**: Template for new robot embodiments (non-pretrained)

During fine-tuning, only the action head corresponding to the specified embodiment tag is trained. Other embodiment heads remain frozen.

### Data Configuration System

The `gr00t/experiment/data_config.py` file defines `BaseDataConfig` subclasses that specify:
- Video modality keys (camera views to use)
- State modality keys (proprioceptive state dimensions)
- Action modality keys (action dimensions)
- Language modality keys (instruction/annotation sources)
- Observation and action temporal indices (history and future)
- Transform pipeline configuration

Built-in data configs are registered in `DATA_CONFIG_MAP`. Users can also define external configs using the format `"module_path:ClassName"` (e.g., `"my_config:RobotConfig"`).

### LeRobot Compatible Data Schema

GR00T uses an extended version of the [LeRobot V2.0 format](https://github.com/huggingface/lerobot). Key additions:

1. **`meta/modality.json`** (required): Maps concatenated state/action arrays to semantic fields with metadata:
   - `start`/`end`: Array slice indices
   - `rotation_type`: Format for rotation representations (quaternion, euler, etc.)
   - `absolute`: Whether action is absolute or delta
   - `range`: Min/max values for the modality

2. **Multiple annotation channels**: Support for `annotation.<source>.<type>.<name>` fields (e.g., `annotation.human.action.task_description`)

3. **Video standardization**: Remap camera names via `"video": {"<new_key>": {"original_key": "<old_key>"}}`

See `getting_started/LeRobot_compatible_data_schema.md` for detailed schema documentation.

## Key Implementation Details

### Model Components and Fine-Tuning

The `GR00T_N1_5` class (`gr00t/model/gr00t_n1.py`) has several components that can be independently fine-tuned:

- `tune_visual`: Fine-tune vision encoder (expensive, only if visual domain differs significantly)
- `tune_llm`: Fine-tune language model (rarely needed, frozen by default in N1.5)
- `tune_projector`: Fine-tune MLP projector (enabled by default)
- `tune_diffusion_model`: Fine-tune shared DiT transformer (disabled by default to save memory)

Embodiment-specific action heads are automatically selected based on the dataset's `EmbodimentTag` and fine-tuned while others remain frozen.

### Inference Flow

1. Load policy: `Gr00tPolicy` wraps the model and handles transforms
2. Process observation: Apply `modality_transform` to raw observations
3. Model forward: VLM processes vision+language → Projector → DiT denoises actions
4. Denormalize: Apply inverse transforms to get executable actions
5. Execute: Send actions to robot controller

The action head generates action chunks (multiple timesteps) per inference call for temporal consistency.

### Training Loop

Training uses PyTorch Lightning via `gr00t/experiment/trainer.py` and `gr00t/experiment/runner.py`:

1. Load datasets with `LeRobotSingleDataset` or `LeRobotMixtureDataset`
2. Apply data transforms
3. Sample batches with temporal context (observation history + action future)
4. Compute losses:
   - Flow matching loss (main action prediction loss)
   - FLARE loss (optional, for video-based training)
5. Backprop through selected trainable components
6. Log to wandb (if enabled)

### Testing Infrastructure

Tests are in `tests/`:
- `test_dataset.py`: Dataset loading and modality parsing
- `test_load.py`: Model loading and checkpoint compatibility
- `test_load_video.py`: Video decoding with different backends

CI runs format checks (isort, black), linting (ruff), and pytest on every PR.

## Common Workflows

### Adding a New Robot Embodiment

1. Collect demonstration data (video, state, action)
2. Convert to LeRobot format with `meta/modality.json`
3. Create a custom `BaseDataConfig` subclass defining modalities and transforms
4. Use `EmbodimentTag.NEW_EMBODIMENT` for fine-tuning
5. Fine-tune with `--embodiment-tag new_embodiment --data-config your_config`

See `getting_started/3_0_new_embodiment_finetuning.md` for detailed instructions.

### Debugging Dataset Issues

1. Use `scripts/load_dataset.py` to inspect data loading
2. Check `meta/modality.json` schema matches parquet column indices
3. Verify video files exist in `videos/chunk-*/<camera_name>/episode_*.mp4`
4. Check `meta/episodes.jsonl` and `meta/tasks.jsonl` are valid

### Performance Benchmarks

**Inference (H100, single sample)**:
- VLM Backbone: 23.18 ms
- Action Head (4 diffusion steps): 4 × 6.18 ms = 24.7 ms
- Full model: ~48 ms (20 Hz)

**Fine-tuning Hardware**:
- Optimal: H100 or L40 (full model fine-tuning)
- Budget: 2× A6000 or 2× RTX 4090 (LoRA fine-tuning)
- Minimum: RTX 4090 (with `--no-tune_diffusion_model`)

**TensorRT Deployment (Thor, FP16/FP8)**:
- VLM ViT: 5.21 ms (FP16) → 4.10 ms (FP8)
- VLM LLM: 8.53 ms (FP16) → 5.81 ms (FP8/FP4)
- Action DiT: 5.46 ms (FP16) → 3.41 ms (FP8)

## Important Files and Locations

- **Entry points**: `scripts/` (inference_service.py, gr00t_finetune.py, eval_policy.py)
- **Model core**: `gr00t/model/` (policy.py, gr00t_n1.py, action_head/, backbone/)
- **Data pipeline**: `gr00t/data/` (dataset.py, schema.py, transform/, embodiment_tags.py)
- **Training**: `gr00t/experiment/` (trainer.py, runner.py, data_config.py)
- **Deployment**: `deployment_scripts/` (export_onnx.py, trt_*.py)
- **Documentation**: `getting_started/` (Jupyter notebooks and markdown guides)
- **Examples**: `examples/` (Libero, SimplerEnv, SO-100, RoboCasa benchmarks)

## Additional Resources

- **Getting Started**: `getting_started/` folder contains 6 tutorials (dataset loading, inference, fine-tuning, new embodiments, deep dive, deployment)
- **Reference Architecture**: `reference_architecture/reference_architecture.md`
- **Paper**: https://arxiv.org/abs/2503.14734
- **Model Checkpoint**: https://huggingface.co/nvidia/GR00T-N1.5-3B
- **Dataset**: https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
