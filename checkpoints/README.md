# Checkpoints

Put the trained model weights in this folder so the Streamlit app can load
them for the **Live Demo** page.

## Expected file names

| Model          | File name                 | Produced by                                  |
| -------------- | ------------------------- | -------------------------------------------- |
| Custom CNN     | `CustomCNN_best.pth`      | `Anti_Spoofing_Skenario_1.ipynb`             |
| ResNet18       | `ResNet18_best.pth`       | `Anti_Spoofing_Skenario_1.ipynb`             |
| MobileNetV2    | `MobileNetV2_best.pth`    | transfer-learning training notebook          |
| EfficientNetB0 | `EfficientNetB0_best.pth` | transfer-learning training notebook          |

Any `*.pth` / `*.pt` / `*.ckpt` file is detected as long as its name contains
one of: `customcnn`, `baselinecnn`, `cnnmodel`, `mobilenet`, `resnet18`,
`efficientnet`.

## How to get the files

1. Run the training notebook (Google Colab).
2. The notebook downloads the best checkpoints automatically:

   ```python
   from google.colab import files
   files.download("/content/model_checkpoints/CustomCNN_best.pth")
   files.download("/content/model_checkpoints/ResNet18_best.pth")
   ```

3. Move the downloaded `*_best.pth` files into this `checkpoints/` folder.

> Checkpoints are **not** committed to git (see `.gitignore`).
