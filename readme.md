# FairDI: Fair Distillation for Equitable Medical Image Screening

## The Core Contribution (TL;DR)
This repository implements the **FairDI** (Fairness Distillation) pipeline to train equitable computer vision models for medical diagnosis. It addresses demographic bias by using a **Fair Identity Scaling (FIS) loss** (which leverages Optimal Transport to scale gradients for underrepresented groups) and a specialized Teacher-Student distillation framework. The pipeline trains a fair backbone, specializes teacher models per demographic group, and finally distills this knowledge into a single, highly performant and fair student model.

## Environment & Hardware Setup
This codebase is designed for high-throughput training using PyTorch 2.0+ features (`torch.compile`) and automatic mixed precision (AMP). 

- **Hardware Requirements:** 
  - An NVIDIA GPU with CUDA support is highly recommended. 
  - **VRAM Note:** The default batch size is `256` (configured in `config.py`), which caches features and logits heavily in memory. If you are running on consumer GPUs (e.g., RTX 3090/4090 with 24GB VRAM) and encounter Out-Of-Memory (OOM) errors, drop the `batch_size` to `128` or `64`.
- **Dependencies:**
  - Python 3.8+
  - PyTorch 2.0+
  - See `requirements.txt` for the full list (includes `timm`, `fairlearn`, `transformers`, etc.).

**Quick Setup:**
```bash
# Clone the repository (replace with actual URL)
git clone https://github.com/your-username/daignose.git
cd daignose

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation
The repository supports multiple medical imaging datasets: **HAM10000**, **OL3I**, **FITZPATRICK17K**, **PAPILA**, **HARVARD_GF**, and **OASIS**. 

Downloading, preprocessing, and splitting are handled automatically by the respective dataset scripts (e.g., `ham10000.py`, `oasis.py`). The dataset scripts manage the splits internally to strictly prevent data leakage across train/val/test boundaries.

**Steps to configure data:**
1. Open `config.py`.
2. Set `Config.data_path` to your base dataset directory.
3. Choose the dataset by modifying `Config.dataset_config['name']`.
4. Define the sensitive attribute for fairness grouping (e.g., `'Sex_binary'`, `'Skin_type_binary'`, etc.) using `Config.dataset_config['sensitive_attr']`.

## Reproducing the Results
Running the full pipeline is straightforward. The entire workflow—backbone training, teacher specialization, and student distillation—is executed sequentially in one shot. Features and teacher logits are automatically cached in memory to massively speed up the distillation phase.

Run the training and evaluation pipeline:
```bash
python main.py
```

*What exactly does this script do?*
1. **Backbone Training:** A baseline model (`efficientnet_v2_s` by default) is trained using the FIS loss.
2. **Feature Caching:** The backbone's features are extracted and cached to avoid redundant forward passes.
3. **Teacher Training:** Specialized teacher models are trained for *each* demographic group defined in your dataset.
4. **Logit Caching:** The teacher outputs are cached.
5. **Student Distillation:** A final student model is trained to mimic the teachers, using knowledge distillation combined with the FIS loss.

Evaluation results (Fairness metrics, AUC, etc.) are printed directly to the console. The final models (backbone, teachers, and student) and the evaluation metrics are automatically saved to the `saved_models/` directory.

## Project Structure
A quick map to help you navigate the codebase:

- **`main.py`**: The primary entry point. Initializes the dataset and triggers the pipeline.
- **`config.py`**: The central brain for hyperparameter tuning. Adjust batch sizes, learning rates, CutMix probabilities, and AMP settings here.
- **`run_full_pipeline.py`**: The core workflow script linking backbone, teachers, and student training. Look here for the caching mechanisms.
- **`fis_loss.py`**: The custom PyTorch implementation of the **Fair Identity Scaling (FIS) loss**. Includes the 1D Wasserstein/Optimal Transport distance calculations.
- **`*_trainer.py`**: `backbone_trainer.py`, `teacher_trainer.py`, and `student_trainer.py` contain the raw PyTorch training loops for each phase.
- **Dataset Scripts**: `ham10000.py`, `oasis.py`, `fitzpatrick17k.py`, etc., contain the dataset-specific downloading, parsing, and splitting logic.

## Citation
If you find this code or our methodology useful for your research, please cite our paper:

```bibtex
@article{your_paper_placeholder,
  title={FairDistillation: Equitable Medical Image Screening via Teacher-Student Distillation and Fair Identity Scaling},
  author={First Author and Co-authors},
  journal={Placeholder Journal},
  year={202X}
}
```