# DIAGNOSE: Fair Distillation for Equitable Medical Image Screening

## The Core Contribution (TL;DR)
This repository implements the **DIAGNOSE** (Fairness Distillation) pipeline to train equitable computer vision models for medical diagnosis. It addresses demographic bias by using a **Fair Identity Scaling (FIS) loss** (which leverages Optimal Transport to scale gradients for underrepresented groups) and a specialized Teacher-Student distillation framework. The pipeline trains a fair backbone, specializes teacher models per demographic group, and finally distills this knowledge into a single, highly performant and fair student model.

## Environment & Hardware Setup
This codebase uses PyTorch 2.0+ features (`torch.compile`) and automatic mixed precision (AMP).

**Key dependencies:**
- Python 3.8+
- PyTorch 2.0+
- timm
- transformers
- fairlearn

**Quickstart:**
```bash
# Clone the repository (replace with actual URL)
git clone https://github.com/your-username/daignose.git
cd daignose

# Install dependencies
pip install -r requirements.txt
```

After installing dependencies, update `config.py` to set the data root and select the dataset. The key fields are `Config.data_path`, `Config.dataset_config['name']`, and `Config.dataset_config['sensitive_attr']`.

### Dataset
Due to the data use agreements, we cannot directly share the download link. Please register and download datasets using the links from the table below:

| **Dataset**  | **Access**                                                                                    |
|--------------|-----------------------------------------------------------------------------------------------|
| CheXpert     | https://stanfordmlgroup.github.io/competitions/chexpert/                                      |
| OL3I         | https://stanfordaimi.azurewebsites.net/datasets/3263e34a-252e-460f-8f63-d585a9bfecfc          |
| PAPILA       | https://www.nature.com/articles/s41597-022-01388-1#Sec6                                       |
| HAM10000     | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T               |
| Oasis-1      | https://www.oasis-brains.org/#data                                                            |
| Fitzpatrick17k | https://github.com/mattgroh/fitzpatrick17k                                                  |
| Harvard-GF3300  |  https://ophai.hms.harvard.edu/datasets/harvard-glaucoma-fairness-3300-samples/            |

## Citation
If you find this code or our methodology useful for your research, please cite our paper:

```bibtex
@article{your_paper_placeholder,
  title={DIAGNOSEstillation: Equitable Medical Image Screening via Teacher-Student Distillation and Fair Identity Scaling},
  author={First Author and Co-authors},
  journal={Placeholder Journal},
  year={202X}
}
```