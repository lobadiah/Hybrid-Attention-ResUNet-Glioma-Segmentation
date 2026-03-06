# Hybrid Attention ResUNet Glioma Segmentation

## Overview  
This repository implements five models for medical image segmentation of glioma tumors. The models include:
1. Baseline U-Net  
2. U-Net with Dice Loss  
3. ResNet U-Net  
4. Attention U-Net  
5. 3D U-Net  

## Installation  
To set up the environment:
1. Clone the repository:
   ```bash
   git clone https://github.com/lobadiah/Hybrid-Attention-ResUNet-Glioma-Segmentation.git
   cd Hybrid-Attention-ResUNet-Glioma-Segmentation
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Examples  
### Baseline U-Net  
- To train the Baseline U-Net model:
  ```bash
  python train_unet.py --model baseline
  ```
- To predict using the Baseline U-Net model:
  ```bash
  python predict.py --model baseline --input <image_path> --output <output_path>
  ```

### U-Net with Dice Loss  
- Training:
  ```bash
  python train_unet.py --model unet_dice
  ```

### ResNet U-Net  
- Training:
  ```bash
  python train_unet.py --model resnet_unet
  ```

### Attention U-Net  
- Training:
  ```bash
  python train_unet.py --model attention_unet
  ```

### 3D U-Net  
- Training:
  ```bash
  python train_unet.py --model unet_3d
  ```

## Model Comparison  
| Model                  | IOU Score | Dice Score | Training Time |
|-----------------------|-----------|------------|---------------|
| Baseline U-Net       | X.XX      | X.XX       | XX hours      |
| U-Net with Dice Loss | X.XX      | X.XX       | XX hours      |
| ResNet U-Net         | X.XX      | X.XX       | XX hours      |
| Attention U-Net      | X.XX      | X.XX       | XX hours      |
| 3D U-Net             | X.XX      | X.XX       | XX hours      |

## Training Guide  
- Adjust the parameters in the `config.py` file as per your requirements.
- Ensure that the dataset is properly organized and accessible by the scripts.
- Follow the script instructions for commands.

## Evaluation Metrics  
- Intersection over Union (IoU)  
- Dice Coefficient  
- Precision  
- Recall  

## Debugging Guide  
- Check the logs generated during training for issues.
- Ensure that the input data is preprocessed correctly.
- For common errors, refer to the issues section on GitHub.

## Results  
The results of the models can vary based on the dataset and training configurations. Experiment with different settings for optimal performance.
