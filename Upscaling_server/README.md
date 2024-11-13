# Real-ESRGAN

This is a forked version of Real-ESRGAN. This repo includes detailed tutorials on how to use Real-ESRGAN on Windows locally through the .exe or PyTorch for both images and videos. 

# This version of Real-ESRGAN is out of date. The main branch has now officially support Windows, go [here](https://github.com/xinntao/Real-ESRGAN) to the main branch. You can still use this repo as a reference for setting up environment and such.

[![download](https://img.shields.io/github/downloads/xinntao/Real-ESRGAN/total.svg)](https://github.com/xinntao/Real-ESRGAN/releases)
[![Open issue](https://isitmaintained.com/badge/open/xinntao/Real-ESRGAN.svg)](https://github.com/xinntao/Real-ESRGAN/issues)
[![LICENSE](https://img.shields.io/github/license/xinntao/Real-ESRGAN.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)
[![python lint](https://github.com/xinntao/Real-ESRGAN/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/.github/workflows/pylint.yml)

1. [Colab Demo](https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo) for Real-ESRGAN <a href="https://colab.research.google.com/drive/1k2Zod6kSHEvraybHl50Lys0LerhyTMCo?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>.
2. [Portable Windows executable file](https://github.com/xinntao/Real-ESRGAN/releases). You can find more information [here](#Portable-executable-files).

Real-ESRGAN aims at developing **Practical Algorithms for General Image Restoration**.<br>
We extend the powerful ESRGAN to a practical restoration application (namely, Real-ESRGAN), which is trained with pure synthetic data.

:triangular_flag_on_post: **Updates**

- :white_check_mark: [The inference code](inference_realesrgan.py) supports: 1) **tile** options; 2) images with **alpha channel**; 3) **gray** images; 4) **16-bit** images.
- :white_check_mark: The training codes have been released. A detailed guide can be found in [Training.md](Training.md).

### :book: Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

> [[Paper](https://arxiv.org/abs/2107.10833)] &emsp; [Project Page] &emsp; [Demo] <br>
> [Xintao Wang](https://xinntao.github.io/), Liangbin Xie, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> Applied Research Center (ARC), Tencent PCG<br>
> Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences

<p align="center">
  <img src="assets/teaser.jpg">
</p>

---

Any updates on the main repository will not be updated here. Please use this just as a tutorial reference, and refer any new updates from the original. 

There are 2 options to run Real-ESRGAN:
1. [Windows Executable Files (.exe)](https://github.com/bycloudai/Real-ESRGAN-Windows/tree/master#windows-executable-files-exe-vulkan-ver)
2. [CUDA & PyTorch](https://github.com/bycloudai/Real-ESRGAN-Windows/tree/master#cuda--pytorch)

---

## Windows Executable Files (.exe) VULKAN ver. 
(1:4 ratio against CUDA, time it takes VULKAN to run 1 image, CUDA can run 4 images)

Does **not** require a NVIDIA GPU.

You can download **Windows executable files** from https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRGAN-ncnn-vulkan-20210725-windows.zip

This executable file is **portable** and includes all the binaries and models required. No CUDA or PyTorch environment is needed.<br>

1. Place the images in the same folder level as your .exe file. 
2. `cd` To where your file is located on your command prompt, and you can simply run the following command:
   (replace the <> with the corresponding name)
   ```bash
   realesrgan-ncnn-vulkan.exe -i <input_image> -o output.png
   ```
3. (Optional) run through a video 

   I've wrote a simple Python file that would generate a .bat file that will help you run through all the frames in a video. Download the `func.py` in this repo.
   
   Open up [Anaconda](https://www.anaconda.com/download/) prompt, input these commands to download these libraries:
   ```
   pip install opencv-python
   conda install -c conda-forge ffmpeg
   ```
   
   Create a folder called "📂input_videos" and drop the video inside this folder.
   ```
   📂Real-ESRGAN-Master/
    ├── 📂input_videos/
    │   └── 📜your_video.mp4 <--
   ```
   
   Run the following command in anaconda prompt: (replace the <>)
   ```
   python func.py <your_video_file>
   ```
   
   And after everything is done, you can find your result under the name `<your_video_name>_result.mp4`




Note that it may introduce block inconsistency (and also generate slightly different results from the PyTorch implementation), because this executable file first crops the input image into several tiles, and then processes them separately, finally stitches together.

This executable file is based on the wonderful [Tencent/ncnn](https://github.com/Tencent/ncnn) and [realsr-ncnn-vulkan](https://github.com/nihui/realsr-ncnn-vulkan) by [nihui](https://github.com/nihui).

---

## CUDA & PyTorch 
(1:4 ratio against CUDA, time it takes VULKAN to run 1 image, CUDA can run 4 images)

Requires a NVIDIA GPU

- Download [Anaconda](https://www.anaconda.com/download/)

### Installation

1. Clone repo 
    Either download this repo manually through the download button on the top right, 
    
    ```bash
    git clone https://github.com/xinntao/Real-ESRGAN.git
    ```
    and enter the folder with the command
    
    ```bash
    cd <your_file_path>/Real-ESRGAN
    ```

2. Install dependent packages

    ```bash
    conda create -n RESRGAN python=3.7
    conda activate RESRGAN #activate the virtual environment
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    pip install basicsr
    pip install -r requirements.txt
    ```
    
3. Download pre-trained models
   
   Download pre-trained models here: [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)
   
   and put it in experiments/pretrained_models
   
   ```
   📂Real-ESRGAN-Master/
    ├── 📂experiments/
    │   └── 📂pretrained_models/
    │       └── 📜RealESRGAN_x4plus.pth
    │   
   ```

4. Inference to obtain image results!
   Drag and drop any images into the "📂inputs" folder, and run the following command:
   ```bash
   python inference_realesrgan.py --model_path experiments/pretrained_models/RealESRGAN_x4plus.pth --input inputs
   ```
   You can find your results in the "📂results" folder!
   
5. (optional) Inference to obtain video results!
   
   If you want to upscale a video, you will have to manually seperate the video into images with FFMPEG
   
   So first install ffmpeg:
   ```
   conda install -c conda-forge ffmpeg
   ```
   Then you drag and drop your video into the base folder. which is inside "📂Real-ESRGAN-Master" and on the same level with "📂experiments". 
   
   convert your video into png with the following command. replace out <> with the video name.
   ```
   ffmpeg -i <your_video.format, eg: video.mp4> inputs/<video_name>%d.png
   ```
   
   Run the AI
   ```bash
   python inference_realesrgan.py --model_path experiments/pretrained_models/RealESRGAN_x4plus.pth --input inputs
   ```
   
   Replace the details in <> and run this command
   ```bash
   ffmpeg -i results/<video_name>%d.png -c:v libx264 -vf fps=<your original video's FPS> -pix_fmt yuv420p <video_name>_result.mp4
   ```
  You will see your video is now upscaled x4 and can be found under the name `<video_name>_result.mp4`
  
  Remember to delete all the images inside "📂inputs" if you want to run on another video or image.



## :computer: Training

A detailed guide can be found in [Training.md](Training.md).

## BibTeX

    @Article{wang2021realesrgan,
        title={Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
        author={Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
        journal={arXiv:2107.10833},
        year={2021}
    }

## :e-mail: Contact

If you have any question, please join the discord channel: https://dsc.gg/bycloud.
