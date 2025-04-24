# Ring Segmentation using Mask R-CNN

## Overview

This repository contains a complete system for ring segmentation using Mask R-CNN. The project involves training a deep learning model on a ring segmentation dataset, extracting ring frames, and generating 3D models of the rings using a mixed depth and scale model. The system is designed to work with images extracted from videos, enabling automated ring segmentation and 3D reconstruction.

## Project Structure

The project is divided into three main parts:

1. **Ring-image-segmentation.ipynb**: This notebook contains the Mask R-CNN model for ring segmentation.
2. **RingFrameExtraction.ipynb**: This notebook extracts frames from videos, preparing them for segmentation.
3. **MiDaS.ipynb**: This notebook applies the MiDaS model for depth estimation, which is then used to generate 3D models of the rings.

## Requirements

The following libraries are required to run the notebooks:

- `torch`
- `torchvision`
- `roboflow`
- `pycocotools`
- `albumentations`
- `opencv-python`
- `matplotlib`
- `PIL`
- `numpy`
- `cv2`

You can install the required dependencies using pip.

## Dataset from Roboflow

The dataset used for training and evaluation is sourced from **Roboflow**. You can access and download the dataset from the following link:

- [Ring Segmentation Dataset on Roboflow](https://universe.roboflow.com/postcards-c0lec/ring-segmentation-zdria/dataset/2)

You can either use the pre-trained model or upload your custom dataset to Roboflow for training.

## How to Run the Code

### Step 1: Train the Mask R-CNN Model (Ring Segmentation)

The first step is to train the **Mask R-CNN** model for ring segmentation. Use the `Ring-image-segmentation.ipynb` notebook to train the model using the Roboflow dataset. The notebook will guide you through:

- Loading and processing the dataset.
- Training the Mask R-CNN model on the ring segmentation task.
- Saving the trained model for future inference. [Link for trainned model](https://drive.google.com/drive/folders/1o9ueta73Bnf_ahY7Bu1DnabO1RIOezHK?usp=drive_link)

### Step 2: Extract Frames from Videos

Once the Mask R-CNN model is trained, use the `RingFrameExtraction.ipynb` notebook to extract frames from a video file. This notebook will:

- Read video files: [link of sample video](https://drive.google.com/file/d/1DrobI9aun48xQ6sAGBH2jxLHjm2ASFV-/view?usp=drive_link)
- Extract frames at specified intervals.
- Save extracted frames as images for further processing.[generated images](https://drive.google.com/drive/folders/1FGjOaJkgJbyncHMtRdLx4ucc2svufGZw?usp=drive_link)

### Step 3: Generate 3D Models

Finally, the `MiDaS.ipynb` notebook uses the MiDaS model to estimate depth from the segmented frames. Depth estimation is used to generate 3D models of the rings. This step involves:

- Using the trained Mask R-CNN model to segment the rings.
- Applying the MiDaS depth model to generate depth maps.
- Using the depth maps to create 3D models.

## Inference

Once the model is trained and frames are extracted, you can use the trained Mask R-CNN model to perform inference on new images or video frames. The `Ring-image-segmentation.ipynb` notebook provides functionality for running inference and visualizing the segmented rings.

## Conclusion

This repository provides an end-to-end pipeline for ring segmentation, frame extraction, and 3D model generation from videos. The system utilizes Mask R-CNN for object detection and segmentation, MiDaS for depth estimation, and custom post-processing steps for generating 3D models.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
