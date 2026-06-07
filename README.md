# rsgen

**rsgen** is a road scene image generation module that fine-tunes [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) on driving scene data. It converts structured scene graphs into natural-language captions and uses them to generate realistic road scene images. It is designed to plug into a semantic communications pipeline (GBSED) where scene graphs are transmitted wirelessly and images are reconstructed at the receiver from the decoded scene graph semantics.

![System Architecture](Images/architecture/Sys_model.png)

---

## Overview

The system operates end-to-end:

1. **Scene graph → text**: structured driving scene graphs (from [roadscene2vec](https://github.com/AICPS/roadscene2vec)) are converted to natural-language captions describing spatial relationships between road actors (cars, pedestrians, lanes, etc.).
2. **Fine-tuned Stable Diffusion**: a Stable Diffusion UNet is fine-tuned on paired (caption, image) samples from the driving dataset, freezing the VAE and text encoder.
3. **Wireless transmission simulation**: the original images are also passed through a physical-layer channel model (OFDM MIMO CDL) provided by the parent GBSED pipeline for comparison.
4. **End-to-end pipeline**: the `phy_layer_enhancer` class orchestrates scene graph encoding/transmission, caption generation, image generation, and direct image transmission, writing results to disk and logging timing.

---

## Repository Structure

```
rsgen/
├── Config/
│   ├── sd_finetuning.yaml              # Training & model configuration
│   └── sd_train_dataset_extraction.yaml # Scene graph extraction & relation mapping
├── Images/
│   ├── Initial/                        # Original driving images
│   ├── Transmitted/                    # Images after wireless channel transmission
│   ├── Generated/                      # Images generated from scene graph captions
│   └── architecture/                   # System architecture diagram
├── pipeline/
│   └── pipeline.py                     # End-to-end phy_layer_enhancer pipeline
├── text2image/
│   └── sd_trainer.py                   # Stable Diffusion fine-tuning & inference
├── utils/
│   ├── datasetGenerator.py             # sg2text conversion + dataset classes
└── requirements.txt
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

The project also depends on **GBSED** (Graph-Based Semantic Encoder-Decoder), which must be cloned and available at `../../gbsed` relative to this repository.

---

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
  --extraction_filename ../../gbsed/Config/pipeline_extraction.yaml \
  --learning_filename ../../gbsed/Config/pipeline_learning.yaml \
  --sd_config_filename Config/sd_finetuning.yaml \
  --com_model_endpoint ../../gbsed/Communication/weights/neural_rx_ofdm_mimo_cdl_final.h5 \
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

- [roadscene2vec](https://github.com/AICPS/roadscene2vec) for scene graph extraction
- [Stable Diffusion](https://arxiv.org/abs/2112.10752) (Rombach et al., 2022)
- [Sionna](https://nvlabs.github.io/sionna/) for the wireless channel simulation

---

*Author: Phil Polo*
