# 🏞️ Gen4Gen: Generative Data Pipeline for Generative Multi-Concept Composition (BMVC 2025)

<a href="https://danielchyeh.github.io/Gen4Gen/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue" height=20.5></a>
<a href="https://arxiv.org/abs/2402.15504"><img src="https://img.shields.io/static/v1?label=Paper&message=Link&color=green" height=20.5></a>
<a href=""><img src="https://img.shields.io/static/v1?label=Project&message=Video&color=red" height=20.5></a>

By [Chun-Hsiao Yeh*](https://danielchyeh.github.io/), [Ta-Ying Cheng*](https://ttchengab.github.io/), [He-Yen Hsieh*](https://www.linkedin.com/in/he-yen-hsieh/), [Chuan-En Lin](https://chuanenlin.com/), [Yi Ma](https://people.eecs.berkeley.edu/~yima/), [Andrew Markham](https://www.cs.ox.ac.uk/people/andrew.markham/), [Niki Trigoni](https://www.cs.ox.ac.uk/people/niki.trigoni/), [H.T. Kung](https://www.eecs.harvard.edu/htk/), [Yubei Chen](https://yubeichen.com/) ( * equal contribution)

> UC Berkeley, University of Oxford, Harvard University, CMU, HKU, UC Davis

###### tags: `stable diffusion` `personalized text-to-image generation` `llm`

This repo is the official implementation of "**Gen4Gen: Generative Data Pipeline for Generative Multi-Concept Composition**".

<br>
<div class="gif">
<p align="center">
<img src='assets/CVPR24-Gen4Gen-animation-HowItWorks.gif' align="center" width=800>
</p>
</div>

**TL;DR**: We introduce a dataset creation pipeline, **Gen4Gen**, to compose personal concept into realistic scenes with complex compositions, accompanied by detailed text descriptions.

## 📝 Updates

>- [July 26 2025] [⚡️NEWS⚡️] Gen4Gen is accepted to BMVC 2025 @ UK!" 
>- [Feb 26 2024] [⚡️NEWS⚡️] Added  🏞️ Gen4Gen: dataset creation pipeline code!" 
>- [Feb 15 2024] Added demo video 🔥 for "How Gen4Gen Works" 

## 🔎 Overview of This Repository

- [**gen4gen**](gen4gen/): Toolkits for Gen4Gen pipeline.

## 🎁 Prepare Personal Assets

Please prepare your personal images and put them under `data/s0_source_images`. Our personal images are from [Unsplash](https://unsplash.com/license). The structure of <a href="#2">🗂 `data/s0_source_images`</a><br> looks like this:

<details>
<summary><a name="2"></a>🗂 Structure of data/s0_source_images </summary>

```shell
../data/s0_source_images
└── cat_dog_houseplant_3objs
    ├── cat
    │   └── sergey-semin-agQhOHQipoE-unsplash.jpg
    ├── dog
    │   ├── Copy of 5.jpeg
    │   └── Copy of 6.jpeg
    └── houseplant
        ├── Copy of 1.png
        ├── Copy of 2.png
        ├── Copy of 3.png
        └── Copy of 5.png
└── [folder_of_other_scenes]
    ├── [object_name_1]
    │   ├── [image_name_1.jpg]
    │   ├── ...
    │   └── [image_name_n.jpeg]
    ...
    └── [oject_name_n]
```
</details>

## :world_map: <a name="3"></a> Environments
<details>
<summary>Library versions</summary>
    
- <a href="https://pytorch.org/get-started/previous-versions/#:~:text=conda%20install%20pytorch%3D%3D2.1.0%20torchvision%3D%3D0.16.0%20torchaudio%3D%3D2.1.0%20pytorch%2Dcuda%3D12.1%20%2Dc%20pytorch%20%2Dc%20nvidia" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="15" height="15"/> </a> Framework and environment
    - pytorch: 1.13.1
    - cuda: 11.7
    - torchvision: 0.14.1  
- <a href="https://huggingface.co/blog/stable_diffusion" target="_blank" rel="noreferrer"> </a> 🧨 Go-to library for diffusion models
    - diffusers: 0.21.4 
- <a href="https://platform.openai.com/docs/models/overview" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg" alt="OpenAI" width="15" height="15"/> </a> OpenAI API for LLM-Guide Object Composition
    - openai: 0.28.1
- <a href="https://docs.conda.io/en/latest/miniconda.html" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="20" height="20"/> </a> Programming language
    - python: 3.8.5
- <a href="https://developer.nvidia.com/cuda-toolkit" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/sco/2/21/Nvidia_logo.svg" alt="pytorch" width="20" height="20"/> </a> Graphics card
    - **[Gen4Gen pipeline]** GPU: NVIDIA A100-SXM4-40GB x 1 or NVIDIA GeForce RTX 4090-24GB x 1
        >- For Step1, Object Association and Foreground Segmentation, requiring around 2.2GB memory footprint
        >- For Step3, Background Repainting, requring around 17GB memory footprint
    
</details>

## :thumbsup: <a name="10"></a> Acknowledgement   
Our codebase is built based on [DIS](https://github.com/xuebinqin/DIS), [LLM-grounded Diffusion](https://github.com/TonyLianLong/LLM-groundedDiffusion/tree/c35ecb307439834fb4944b5f15116db890be93d9), [SD-XL Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1), and [custom-diffusion](https://github.com/adobe-research/custom-diffusion/tree/4345c288c71e05d29ced32a965be03220841bff0). We really appreciate the authors for the nicely organized code and fantastic works!

## 📬 How to Get Support?
If you have any general questions or need support, please feel free to contact: [Chun-Hsiao Yeh](mailto:daniel_yeh@berkeley.edu), [Ta-Ying Cheng](mailto:taying.cheng@gmail.com) and [He-Yen Hsieh](mailto:m10502103@gmail.com). Also, we encourage you to open an issue in the GitHub repository. By doing so, you not only receive support but also contribute to the collective knowledge base for others who may have similar inquiries.

## :heart: <a name="11"></a> Citation
If you find the codebase and MyCanvas dataset valuable and utilize it in your work, we kindly request that you consider giving our GitHub repository a ⭐ and citing our paper.
```
@misc{yeh2024gen4gen,
  author        = {Chun-Hsiao Yeh and
                   Ta-Ying Cheng and
                   He-Yen Hsieh and
                   David Chuan-En Lin and
                   Yi Ma and
                   Andrew Markham and
                   Niki Trigoni and
                   H.T. Kung and
                   Yubei Chen},
  title         = {Gen4Gen: Generative Data Pipeline for Generative Multi-Concept Composition},
  year          = {2024},
  eprint        = {2402.15504},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```
