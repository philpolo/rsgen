# Towards a Joint Task-Oriented and Generative Semantic Communication Framework for 6G Networks

This repository provides the implementation a joint task-oriented and generative semantic communication architecture that transmits semantic information instead of raw data, enabling both reliable downstream task execution and high-fidelity visual reconstruction under bandwidth-constrained wireless environments.
Unlike conventional communication systems that focus on bit-perfect delivery, the proposed approach preserves the meaning of the transmitted information by operating on semantic scene representations.

![System Architecture](Images/architecture/Sys_model.png)

---

## Overview

The framework consists of two complementary semantic objectives:

. **Task-Oriented Semantic Communication:**: enables downstream inference, such as collision-risk estimation, directly from reconstructed semantic graphs.
. **Generative Semantic Communication:**: reconstructs realistic road-scene images from the recovered semantic representation using diffusion models.
By jointly optimizing these two objectives, the framework supports both machine intelligence and human interpretability.

---

## Repository Structure

```
rsgen/
│
├── Communication/                  # End-to-end physical layer communication framework
│   ├── e2emodel.py                 # OFDM MIMO end-to-end semantic communication model
│   ├── receiver.py                 # Neural receiver implementation
│   └── weights/
│       └── Neural_Demaper.h5       # Pre-trained neural receiver weights
│
├── Config/                         # Configuration files
│   ├── pipeline_extraction.yaml    # Scene graph extraction parameters
│   ├── pipeline_learning.yaml      # Training and inference configuration
│   └── *.yaml                      # Additional experiment configurations
│
├── learning/                       # Learning modules
│   ├── rs2vec_training.py          # RoadScene2Vec model training
│
├── pipeline/                       # Main semantic communication pipeline
│   ├── pipeline.py                 # Entry point of the framework
│   ├── gbsed_core.py               # Core semantic communication engine
│
├── sgautoencoder/                  # Graph semantic encoder/decoder
│   ├── sg_autoencoder.py           # Semantic graph autoencoder
│
├── text2image/                     # Generative semantic decoder
│   ├── sd_trainer.py               # Stable Diffusion interface
│
├── utils/                          # General-purpose utilities
│   ├── datasetGenerator.py         # Dataset creation and preprocessing
│
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── LICENSE                         # License information

```

---

## Dependencies

This project requires Python 3.10+ and a CUDA-capable GPU. Install dependencies with:

```bash
pip install -r requirements.txt
```

Key dependencies include:

- `torch >= 2.8` with CUDA 12.6
- `diffusers >= 0.38`
- `transformers >= 5.9`
- `accelerate`
- `roadscene2vec` (scene graph extraction)
- `sionna >= 1.1` (wireless channel simulation)
- `wandb` (training tracking)
- `detectron2`, `fvcore` (object detection backbone)

## Configuration

### `Config/sd_finetuning.yaml`

Controls training and model paths:

| Parameter | Description |
|---|---|
| `model_configuration.model_name` | HuggingFace model ID (default: `CompVis/stable-diffusion-v1-4`) |
| `model_configuration.checkpoint` | Path to save/load fine-tuned UNet checkpoint |
| `model_configuration.sample_size` | Image resolution (default: `512`) |
| `training_configuration.lr` | Learning rate (default: `1e-6`) |
| `training_configuration.batch_size` | Training batch size (default: `6`) |
| `training_configuration.num_epochs` | Number of training epochs (default: `500`) |
| `datasets.*` | Paths to pre-built train/test/valid pickle datasets |
| `input_datasets.*` | Paths to raw roadscene2vec scene graph datasets |
| `output_dir` | Directory for generated images and captions |

### `Config/sd_train_dataset_extraction.yaml`

Controls scene graph extraction and the relation-to-text mapping used when building captions (e.g. `inDFrontOf` → `"in front of (distant)"`). Refer to the [roadscene2vec documentation](https://github.com/AICPS/roadscene2vec) for full extraction settings.

---

## Usage

### 1. Fine-tune Stable Diffusion

```python
from text2image.sd_trainer import Trainer

trainer = Trainer("Config/sd_finetuning.yaml")
trainer.train()
```

Training logs metrics to Weights & Biases (configure `wandb_configuration` in the YAML).

### 2. Run inference from a scene graph caption

```python
trainer = Trainer("Config/sd_finetuning.yaml")
trainer.load_model(training=False)

caption = "The ego car is near collision with the car. The car is to the right of the ego car."
image, elapsed = trainer.inference(caption)
image.save("output.png")
```

### 3. Run the full end-to-end pipeline

```bash
python pipeline/pipeline.py \
  --extraction_filename ../Config/pipeline_extraction.yaml \
  --learning_filename ../Config/pipeline_learning.yaml \
  --sd_config_filename Config/sd_finetuning.yaml \
  --com_model_endpoint ../Communication/weights/Neural_Demaper.h5  \
  --time_file ../../Data/Outputs/transfer_times.csv
```

This runs scene graph transmission, image generation from captions, and direct image transmission in parallel, saving all results and timing statistics to the output directory.

---

## Scene Graph to Caption Conversion

The `sg2text` class in `utils/datasetGenerator.py` traverses the scene graph adjacency matrix and builds a natural language description:

```python
from utils.datasetGenerator import sg2text
from roadscene2vec.util.config_parser import configuration

config = configuration("Config/sd_train_dataset_extraction.yaml", from_function=True)
converter = sg2text(config)

caption = converter.scene_graph_to_prompt(scene_graph)
# e.g. "The ego car is near collision with the car. The car is to the right of the ego car."
```

---

## Examples

The table below shows original driving images alongside the corresponding images generated by the fine-tuned model from scene graph captions. Each generated image is produced entirely from the natural-language description of the scene — no pixel information from the original is used.

![Examples grid](Images/examples_grid.jpg)

| Original | Generated |
|:---:|:---:|
| <img src="Images/Initial/00004368.jpg" width="80%"/> | <img src="Images/Generated/00004368.jpg"/> |
| <img src="Images/Initial/00056572.jpg" width="80%"/> | <img src="Images/Generated/00056572.jpg"/> |
| <img src="Images/Initial/00098349.jpg" width="80%"/> | <img src="Images/Generated/00098349.jpg"/> |
| <img src="Images/Initial/00098353.jpg" width="80%"/> | <img src="Images/Generated/00098353.jpg"/> |
| <img src="Images/Initial/00098372.jpg" width="80%"/> | <img src="Images/Generated/00098372.jpg"/> |
| <img src="Images/Initial/00634491.jpg" width="80%"/> | <img src="Images/Generated/00634491.jpg"/> |
| <img src="Images/Initial/00634522.jpg" width="80%"/> | <img src="Images/Generated/00634522.jpg"/> |
| <img src="Images/Initial/00634638.jpg" width="80%"/> | <img src="Images/Generated/00634638.jpg"/> |
| <img src="Images/Initial/00634651.jpg" width="80%"/> | <img src="Images/Generated/00634651.jpg"/> |
| <img src="Images/Initial/00634672.jpg" width="80%"/> | <img src="Images/Generated/00634672.jpg"/> |
| <img src="Images/Initial/00635704.jpg" width="80%"/> | <img src="Images/Generated/00635704.jpg"/> |
| <img src="Images/Initial/00635738.jpg" width="80%"/> | <img src="Images/Generated/00635738.jpg"/> |
| <img src="Images/Initial/00662022.jpg" width="80%"/> | <img src="Images/Generated/00662022.jpg"/> |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this work, please also cite:


---

