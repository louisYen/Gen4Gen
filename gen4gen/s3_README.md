# 3️⃣ Step 3: Background Repainting

## 🚩 Current directory path is at `gen4gen`

### 🔖 Quick Note
- For our background images, we download copyright-free sources from [Unsplash](https://unsplash.com/). Once downloaded, these image files are organized within the `backgrounds` folder. The file structure for `backgrounds` is listed in <a href="#0"> Structure of backgorunds</a><br>
- Repainted images are in $1024\times1024$ resolution.

```bash=
$ python s3_background_repainting.py \
    --src-dir <path-from-step2> \
    --bkg-dir <background-image-directories> \
    --dest <path-for-saved-repainted-images> \
    --objects <a-series-object-name>
```

### 👇🏼 A quick run
<details>
<summary>Example</summary>

```bash=
$ python s3_background_repainting.py \
    --src-dir ../data/s2_object_compositions/1_cat+dog+houseplant \
    --bkg-dir backgrounds/garden \
    --dest ../data/s3_background_repainting \
    --objects cat dog houseplant
```
</details>

---
Or we can repaint image using noise as background

```bash=
$ python s3_background_repainting.py \
    --src-dir <path-from-step2> \
    --noise-bkg \
    --dest <path-for-saved-repainted-images> \
    --objects <a-series-object-name>
```

### 👇🏼 A quick run
<details>
<summary>Example</summary>

```bash=
$ python s3_background_repainting.py \
    --src-dir ../data/s2_object_compositions/1_cat+dog+houseplant \
    --noise_bkg \
    --dest ../data/s3_background_repainting \
    --objects cat dog houseplant
```
</details>

> Repainted images are saved to `../data/s3_background_repainting`

---
### 🔠 Arguments

- `--src-dir`: the path of compositions (Please refer to [2️⃣  Step2: LLM-Guided Object Composition](s2_README.md))
- `--dest`: the path for repainted image
- `--noise-bkg`: (Optional) if specified, the starting backgroud image $I_{bg}$ uses noise
- `--bkg-dir`: the path for downloaded copyright-free sources. Our background images are from [Unsplash](https://unsplash.com/).
- `-n-bkg` or `--max-num-bkg-scenes`: randomly select maximum number of background images for each composition (default: 3)
- `-blur-size`: window size of average smoothing on the mask, $\mathcal{M}(I_{fg})$. 
- `--ann-dir`: the path for saving annotations of filename and background prompt in csv format (default: `<your-destination-directory>/gen_annotations`)
- `--guidance-scale`: the parameter for Stable Diffusion XL (SDXL) Inpainting. Please refer to [Guidance-scale](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint#:~:text=Guidance%20scale,from%20the%20prompt)
- `--strength`: the parameter for Stable Diffusion XL (SDXL) Inpainting. Please refer to [Strength](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint#:~:text=Strength,base%20image%20more)


<details>
<summary><a name="0"></a> Structure of backgrounds</span></summary>

```shell
backgrounds
├── garden
    ├── garden1.jpeg
    ├── garden2.jpeg
    └── garden3.jpeg
```
</details>

<p align="right">
      <img src="../assets/Gen4GenLogo.png" alt="Logo" width="" height="55"/>
</p>

<p align="right"><a href="README.md">🔙 Back</a></p>
