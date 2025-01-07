# GuidedHybSensUIR

[Underwater Image Restoration Through a Prior Guided Hybrid Sense Approach and Extensive Benchmark Analysis](https://ieeexplore.ieee.org/document/10824878)
<div>
<span class="author-block">
  Xiaojiao Guo<sup> 👨‍💻‍ </sup>
</span>,
  <span class="author-block">
    <a href='https://cxh.netlify.app/'>Xuhang Chen</a><sup> 👨‍💻‍ </sup>
  </span>,
  <span class="author-block">
    Shuqiang Wang<sup> 📮</sup>
  </span>,
  <span class="author-block">
    <a href='https://cmpun.github.io/'>Chi-Man Pun</a><sup> 📮</sup>
  </span>
  ( 👨‍💻‍ Equal contributions, 📮 Corresponding author)
</div>

<b>University of Macau, SIAT CAS, Huizhou Univeristy, Baoshan Univeristy</b>

In <b>_IEEE Transactions on Circuits and Systems for Video Technology_</b>

# 🔮 Benchmark Dataset

[Kaggle](https://www.kaggle.com/datasets/xuhangc/underwaterbenchmarkdataset)

If you need visualization results, you may contact Dr.Guo via yc27441@um.edu.mo

# ⚙️ Usage

## Training
You may download the dataset first, and then specify TRAIN_DIR, VAL_DIR and SAVE_DIR in the section TRAINING in `config.yml`.

For single GPU training:
```
python train.py
```
For multiple GPUs training:
```
accelerate config
accelerate launch train.py
```
If you have difficulties with the usage of `accelerate`, please refer to <a href="https://github.com/huggingface/accelerate">Accelerate</a>.

## Inference

Please first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TESTING in `config.yml`.

```bash
python infer.py
```

# Citation

```bib
```
