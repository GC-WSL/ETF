<div align="center">
<h1>Expanding then fusing</h1>
<h3>Weakly-supervised remote sensing semantic segmentation via progressive multi-modal fusion</h3>

Guanchun Wang<sup>1</sup>, Xiangrong Zhang<sup>1,*</sup>, Jianxun Lai<sup>1</sup>, Zelin Peng<sup>2</sup>, Tianyang Zhang<sup>1</sup>, Chao Wang<sup>1</sup>, Xu Tang<sup>1</sup>, Licheng Jiao<sup>1</sup>

<sup>1</sup> School of Artificial Intelligence, Xidian University, No. 2, Taibai South Road, Xi'an, 710071, Shannxi, China, <sup>2</sup> School of Computer Science, Shanghai Jiao Tong University, 800 Dongchuan RD. Minhang District, Shanghai, 200240, China

(*) corresponding author.

Accepted by Pattern Recognition ([Paper](https://doi.org/10.1016/j.patcog.2026.113392))

</div>

# Introduction

Weakly supervised semantic segmentation (WSSS) in remote sensing imagery (RSI) offers a cost-effective solution for large-scale land cover mapping by leveraging only image-level labels, also known as the description of category names. Vision-language models (VLMs) have recently exhibited cross-modal alignment capabilities in natural scenes, inspiring recent studies to explore their potential in WSSS, however, their potential in RSI remains untapped. A core challenge lies in adapting VLMs pretrained on natural scenes to RSIs while maintaining the generalization ability. To address this, we propose a progressive multi-modal fusion WSSS framework following an Expanding Then Fusing (ETF) strategy in both vision and text spaces. Specifically, we introduce a Text-space Semantic Expansion module that extends the semantic scope of each remote sensing category via large language models, narrowing the gap between natural and remote sensing concepts. Concurrently, a Vision-space Adjacent Exploration mechanism is introduced to refine pseudo-masks by modeling local spatial coherence through morphological perception, alleviating error accumulation caused by imprecise supervision. The two augmented spaces are then integrated via a Progressive Dual-space Fusion Segmenter that injects the knowledge of expanded textual embeddings into multi-granularity visual features in an iterative manner and obtains precise predictions through the affinity between pixels and texts. Extensive experimental results on three public benchmarks demonstrate that the proposed ETF outperforms previous state-of-the-art methods by 6.54%, 13.02%, and 5.85% mIoU on iSAID, ISPRS Potsdam, and ISPRS Vaihingen, respectively.

<div align=center><img src="img/ETF.png" width="800px"></div>

# Getting Started

## 1. Dataset

### 1.0 Pre-trained CLIP Models

Download the pre-trained CLIP models ([VIT-B-16.pt](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)) and save them to the `pretrained` folder.

### 1.1 Data Preparations

<details>
<summary>
iSAID dataset
</summary>

#### 1. Data Download
You may download the iSAID dataset from their official website: https://captain-whu.github.io/iSAID/dataset.html.

#### 2. Data Preprocessing
After downloading, you may craft your own dataset. Please refer to `tools/isaid.py` for image tiling and the construction process of the iSAID dataset described in https://github.com/ZaiyiHu/CTFA.

#### 3. Pre-processed Data (Optional)
For convenience, we also provide the pre-processed dataset (`iSAID_512_sampled_2.zip`), which already includes the auxiliary masks generated using the CTFA method.
- **Baidu Netdisk Link**: https://pan.baidu.com/s/1WZ2PKze-ihD32yFD64jCvw?pwd=kyuy
- **Extraction Code**: `kyuy`

</details>

<details>

<summary>
ISPRS Potsdam dataset
</summary>

#### 1. Data Download
Datasets for ISPRS Potsdam are widely accessible on the Internet. You may find the original content on: https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx

#### 2. Data Preprocessing
You may refer to tools/potsdam_no_clutter.py for image tiling.


</details>

<details>

<summary>
ISPRS Vaihingen dataset
</summary>

#### 1. Data Download
Datasets for ISPRS Potsdam are widely accessible on the Internet. You may find the original content on: https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx. 

#### 2. Data Preprocessing
You may refer to tools/vaihingen_no_clutter.py for image tiling.
</details>

### 1.2 Structure

Expected directory structure:

```bash
datasets/
    iSAID_512_sampled_2/
        img_dir/
            train/
                ...
            val/
                ...
        ann_dir/
            train/
                ...
            val/
                ...
ETF/
    configs/
    pretrained/
    work_dirs/
    tools/
    ...
```
## 2.Environment

**Python Version**: `>= 3.9` (Recommended: `3.10` or `3.11`)

```bash
# 1. Create a new conda environment
conda create -n eft python=3.9 -y

# 2. Activate the environment
conda activate eft

# 3. Install PyTorch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install other dependencies
pip install -r requirements.txt
```

## 3. Training

### 3.1 Quick Start with the Automated Script

The project provides `run_script.py` in the root directory to streamline training, evaluation (without CRF), and multi‑run averaging.  
By default, it uses the configuration defined by the variable `filename` inside the script (e.g. `configs/isaid2x_sim-msca+cls-erodil-s-mask_test-k7_10k_ctfa.py`).  
You can edit that variable to switch to a different dataset/model (iSAID, Potsdam, Vaihingen).

```bash
cd ETF
python run_script.py 
```

### 3.2 MMSegmentation Tools

If you prefer finer control, you can directly use the standard training and testing scripts.

**Training** (single GPU)
```bash
python tools/train.py configs/vhg_0_sim-msca+cls-erodil-s-mask_test-k11_2k_ome.py
```

**Distributed training** (torchrun example)
```bash
torchrun --nnodes=1 --nproc_per_node=2 tools/train.py configs/psdm_sim-msca+cls-erodil-s-mask_test-k19_3k_ome.py --launcher pytorch
```

## 4. Evaluation

### 4.1 Quick Start with the Automated Script

The project provides `run_evaluation_isaid.py` and `run_evaluation_isprs.py` in the root directory to streamline the inference and CRF post-processing evaluation.  
By default, it uses the configuration defined by the `filename` variable inside the script. You can edit this variable to switch between different datasets/models (iSAID, Potsdam, Vaihingen).

The script supports different evaluation stages via the `--stage` argument:
- `all` (default): Runs the full pipeline—generates probability maps (`.npy`) using `tools/test.py` and then applies CRF post-processing for evaluation.
- `mid`: Only generates the probability maps (useful if you want to inspect intermediate outputs or run custom post-processing later).
- `post`: Skips inference and directly uses existing probability maps to compute the final CRF post-processed results.

```bash
# Run full evaluation (inference + CRF eval)
python eval_script.py --stage all

# Only generate probability maps
python eval_script.py --stage mid

# Only run CRF evaluation on existing maps
python eval_script.py --stage post
```
*Note: The script automatically detects the latest checkpoint (`iter_*?00-*.pth`) in the corresponding `work_dirs/` directory based on the modification time.*

### 4.2 Manual Evaluation (Step-by-Step)

If you prefer finer control, you can run the inference and evaluation steps manually.

**Step 1: Inference (Generate Probability Maps)**  
Use `tools/test.py` with the `--save_infer` flag to generate the probability `.npy` files.
```bash
python tools/test.py configs/isaid2x_sim-msca+cls-erodil-s-mask_test-k7_10k_ctfa.py \
    work_dirs/isaid2x_sim-msca+cls-erodil-s-mask_test-k7_10k_ctfa/iter_10000-adb842dd.pth \
    --save_infer
```

**Step 2: CRF Post-processing and Evaluation**  
Use `tools/eval_crf.py` to evaluate the generated probability maps with CRF post-processing.
```bash
python tools/eval_crf.py \
    --root ../datasets/iSAID_512_sampled_2 \
    --predict-dir work_dirs2025/isaid2x_sim-msca+cls-erodil-s-mask_test-k7_10k_ctfa/prob_npy \
    --num-cls 16 \
    --split test \
    --crf
```
*Note:
 **Potsdam and Vaihingen Datasets**: Add the `--reduce-zero` flag to ignore the zero-class (background/undefined) in the ground truth.*

# Citation
If you find our work useful in your research, please consider citing:
```
@article{wang2026ETF,
    title = {Expanding then fusing: Weakly-supervised remote sensing semantic segmentation via progressive multi-modal fusion},
    author = {Guanchun Wang and Xiangrong Zhang and Jianxun Lai and Zelin Peng and Tianyang Zhang and Chao Wang and Xu Tang and Licheng Jiao},
    journal = {Pattern Recognition},
    volume = {177},
    pages = {113392},
    year = {2026},
}
```
