# MedSAM2
This fork is based on the [MedSAM2 repository from Bowang-lab](https://github.com/bowang-lab/MedSAM2).  

## `inference.py`

The main feature introduced in this version is the `inference.py` script.  
It allows you to run inference to detect Clinical Target Volume (CTV) in CT images using fine-tuned weights.  
For more details about the training process (how to run the training script, dataset format, etc.), please refer to the original repository.

## Installation 

- Create a virtual environment: `conda create -n medsam2 python=3.12 -y` and `conda activate medsam2` 
- Install [PyTorch](https://pytorch.org/get-started/locally/): `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124` (Linux CUDA 12.4)
- Download code `git clone https://github.com/bowang-lab/MedSAM2.git && cd MedSAM2` and run `pip install -e ".[dev]"`
- Download checkpoints: `bash download.sh`
- Optional: Please install the following dependencies for gradio

```bash
sudo apt-get update
sudo apt-get install ffmpeg
pip install gradio==3.38.0
pip install numpy==1.26.3 
pip install ffmpeg-python 
pip install moviepy
```
## How to Reproduce the Results

To fine-tune the model, I followed these steps:

### 1. Preprocess the Dataset

First, I saved all the scripts from the [`utils`](https://github.com/bowang-lab/MedSAM/tree/main/utils) folder in the MedSAM1 repository.  
Inside, there's a README explaining how to organize the data and how to use the `pre_CT_MR.py` script.

### 2. Train the Model

Once the dataset is processed, you can start training. According to the official README:

- Download [`sam2.1_hiera_tiny.pt`](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) and place it in the `checkpoints` folder.
- Add your dataset information in `sam2/configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml` under the `data -> train -> datasets` section.
- Adjust `train_video_batch_size` according to your available GPU memory.

To start training on a single node:

```bash
sh single_node_train_medsam2.sh
- multi-node training
```
With multi node:
```bash
sbatch multi_node_train.sh
```

### 3. Inference
Run the `inference.py` script after setting the appropriate paths at the beginning of the file.
