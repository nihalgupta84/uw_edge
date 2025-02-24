## **🚀 Underwater Image Enhancement**
A **PyTorch implementation** for **underwater image enhancement** using a **transformer-based architecture**.

---

## 📌 **Table of Contents**
- [📥 Setup](#-setup)
- [📂 Dataset Preparation](#-dataset-preparation)
- [🛠️ Training](#️-training)
- [🧪 Testing](#-testing)
- [⚙️ Configuration](#️-configuration)
- [💾 Checkpoints](#-checkpoints)
- [📜 License](#-license)
- [📖 Citation](#-citation)
- [🙏 Acknowledgments](#-acknowledgments)

---

## **📥 Setup**

### **1️⃣ Clone this repository**
```bash
git clone <repository-url>
cd underwater-enhancement
```

### **2️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Set up Weights & Biases (Optional)**
Create a file named **`WANDB_API_KEY.env`** in the root directory and paste your Weights & Biases API key:
```plaintext
WANDB_API_KEY=your_api_key_here
```

---

## **📂 Dataset Preparation**
### **📌 Download Paired Datasets**
The following paired datasets are available (processed and split into a **90/10** training-validation ratio):
- **EVUP**
- **LSUI**
- **UIEB**

**🔗 Download:** [Google Drive Link](https://drive.google.com/drive/folders/1Qn9jf2gtsuLtHZASm-Hms7-HVL1m2OJp?usp=sharing)

### **📌 Dataset Structure**
We provide **two types** of dataset organizations:
1. **Paired Data** (UIEB, EVUP, LSUI) → Contains **raw** images & **reference** images  
2. **Unpaired Data** (UIEB, EVUP, RUIE) → Only **raw** images  

**Example structure for paired datasets:**
```
datasets/
└── UIEB/
    ├── train/
    │   ├── raw/      # Raw underwater images
    │   └── ref/      # Corresponding reference images
    └── val/
        ├── raw/
        └── ref/
```

---

## **🛠️ Training**
### **🚀 Start Training**
```bash
python train.py --config_yaml=config.yml
```

### **🔄 Resume Training**
```bash
python train.py --config_yaml=config.yml TRAINING.RESUME True
```

### **⚙️ Override Configuration Parameters**
You can override **any configuration parameter** via the command line:
```bash
python train.py --config_yaml=config.yml MODEL.NAME edge_v1 OPTIM.BATCH_SIZE 8
```

---

## **🧪 Testing**
### **📷 Run Testing with Reference Images**
```bash
python test.py --config_yaml config.yml MODEL.NAME edge_v1 MODEL.DATASET_NAME "paired or unpai MODEL.SESSION trained_on_uieb TESTING.WEIGHT "checkpoint path" TESTING.VAL_DIR "testing data path"
```
### **📷 Run Testing with No Reference Images**

```bash
python test.py --config_yaml config.yml MODEL.NAME edge_v1 MODEL.DATASET_NAME "paired or unpai MODEL.SESSION trained_on_uieb TESTING.WEIGHT "checkpoint path" TESTING.VAL_DIR "testing data path"  TESTING.INPUT "" TESTING.TARGET ""
```
---

## **⚙️ Configuration**
### **🔹 Configuration Files**
The **`MODEL.SESSION`** parameter in config files defines the project name for **logging and organization**.

| **Config File** | **Description** |
|---------------|----------------|
| `config.yml` | Default configuration |
| `evup.yml` | EVUP dataset-specific settings |
| `guide.yml` | Guide model settings |

---

## **💾 Checkpoints**
During training, **checkpoints** are saved in the directory specified by `TRAINING.SAVE_DIR` in `config.yml`.

| **Checkpoint File** | **Purpose** |
|----------------|-----------------------------------|
| `best_psnr.pth` | Model with highest PSNR metric |
| `best_ssim.pth` | Model with highest SSIM metric |
| `best_loss.pth` | Model with lowest training loss |
| `last_checkpoint.pth` | Last saved model (for resuming training) |

---

## **📜 License**
[Specify License Here]

---

## **📖 Citation**
If you use this repository in your research, please cite:
```bibtex
@article{YourPaper202X,
  author  = {Your Name},
  title   = {Underwater Image Enhancement using Transformers},
  journal = {Journal Name},
  year    = {202X},
  doi     = {XX.XXXX/XXXXX}
}
```

---

## **🙏 Acknowledgments**
We would like to thank:
- **[Contributor 1]** for [specific contribution]
- **[Contributor 2]** for [specific contribution]
- The **research community** for open datasets & resources
