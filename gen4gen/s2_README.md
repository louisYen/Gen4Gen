# 2️⃣ Step 2: LLM-Guided Object Composition

## 🚩 Current directory path is at `gen4gen`
Please run the following command under the `gen4gen` directory.

### 🔖 Quick Note (Before You Run!)
- 🔥🔥 Please remember set your `OPENAI_API_KEY`. We put a placeholder in line 41 of
[`s2_llm_guided_object_composition.py`](s2_llm_guided_object_composition.py#L41) 🔥🔥
- 🔥 Please download [COCO val annotation 2017](http://images.cocodataset.org/annotations/annotations_trainval2017.zip). Once done, make sure you have these two files: `../data/coco/annotations/instances_val2017.json` and `../data/coco/annotations/captions_val2017.json`. 🔥
- We use GPT-3.5 for layout generation and GPT-4 for object ratio generation
    - Object ratio generation is specified by the argument `-obj-ratio` or `--with-object-ratio`. This is an optional argument.

```bash=
$ python s2_llm_guided_object_composition.py \
    --src-dir <path-from-step1> \
    --dest <path-for-saved-compositions> \
    -n <number-of-generated-images-including-multiple-compositions> \
    -ci <output-folder-index> \
    -simg <starting-image-index> \
    --coco-dir <coco-dataset-path> \
    -nipc <number-of-samples-from-each-category-of-coco-dataset> \
    -obj-ratio \
    --objects <a-series-object-names>
```

### 👇🏼 Quick Run
<details>
<summary>Example</summary>

```bash=
$ python s2_llm_guided_object_composition.py \
    --src-dir ../data/s1_segmented_foreground/cat_dog_houseplant_3objs \
    --dest ../data/s2_object_compositions \
    -n 10 \
    -ci 1 \
    -simg 100 \
    --coco-dir ../data/coco/ \
    -nipc 10 \
    -obj-ratio \
    --objects cat dog houseplant
```
</details>

---

### 🔠 Arguments

- `--src-dir`: the path of segmented foreground images (Please refer to [Sec-Select Segmented Foregrounds in Step1](s1_README.md))
- `--dest`: the path for composed image including multiple objects
- `-n` or `--num-samples`: number of generated composed images (default: 10 <- You can also specify this🔥🔥)
- `-ci` or `--composition-index`: the index for the saved folders will start from the specified number you provide (_e.g._, **1**_cat+dog+houseplant)
- `-simg` or `--start-img-id`: the indexing for saved images will initiate from the specified number you provide. (_e.g._, img_000**101**.png)
- `--coco-dir`: if a COCO directory is provided, the bounding boxes and captions of COCO objects serve as the information for LLM guidance.  Otherwise, rely on the [LLM-grounded Diffusion](https://github.com/TonyLianLong/LLM-groundedDiffusion/blob/c35ecb307439834fb4944b5f15116db890be93d9/prompt.py) prompting template for reference.
- `-nipc` or `--num_images_per_category`: the number of COCO paired bounding boxes and captions per category for constructing the LLM prompting template.
- `-obj-ratio` or `--with-object-ratio`: (Optional) if specified, the bounding boxes of objects have a realistic ratio in alignment with the proportions observed in the real world.
- `--objects`: object names


<details>
<summary>Structure of s2_object_compositions</span></summary>

```shell
../data/s2_object_compositions
├── 1_cat+dog+houseplant
│   ├── img_000101.png
│   ├── img_000102.png
│   ├── img_000103.png
│   ├── img_000104.png
│   ├── img_000105.png
│   ├── img_000106.png
│   ├── img_000107.png
│   ├── img_000108.png
│   ├── img_000109.png
│   ├── mask_000101.png
│   ├── mask_000102.png
│   ├── mask_000103.png
│   ├── mask_000104.png
│   ├── mask_000105.png
│   ├── mask_000106.png
│   ├── mask_000107.png
│   ├── mask_000108.png
│   └── mask_000109.png
├── 1_cat+dog+houseplant_directory_classes.json
├── 1_cat+dog+houseplant_params.json
└── 1_cat+dog+houseplant_report.csv
```
</details>

### Troubleshooting
#### IndexError: list index out of range:
```bash=
biggest_obj_bbox = bboxes[biggest_obj_idx]
IndexError: list index out of range
```
> - Reason: This error is triggered when LLM returns inconsistent number of objects we requested.  For example, if we request the layout for a set of objects, such as "cat, dog, and houseplant", GPT might return a layout that includes an unexpected or missing item, such as "cat and houseplant"

<p align="right">
      <img src="../assets/Gen4GenLogo.png" alt="Logo" width="" height="55"/>
</p>

<p align="right"><a href="README.md">🔙 Back</a></p>
