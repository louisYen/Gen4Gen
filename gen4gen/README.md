# GenGen: Generative Data Creation for MyCanvas Dataset

## :hammer_and_wrench: Installation

```bash=
$ conda create --name gen4gen python=3.8.5 -y
$ conda activate gen4gen
$ cd gen4gen
$ pip install -r gen4gen_requirements.txt
$ conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
$ pip install accelerate
```

## Gen4Gen Steps
The generated images have been stored in the `../data` directory at every step. 
We hope to provide a smooth experience for anyone using our code.

### 1️⃣ Step 1: Object Association and Foreground Segmentation

The foreground segmentation instruction is in [s1_README.md](s1_README.md)

### 2️⃣ Step 2: LLM-Guided Object Composition

The LLM-guided object composition instruction is in [s2_README.md](s2_README.md)

### 3️⃣ Step 3: Background Repainting

The background repainting instruction is in [s3_README.md](s3_README.md)


<p align="right"><a href="..">🔙 Back</a></p>
