## Changes:

- Setting e.g. clipmodel = "ViT-L/14@336px" in demo.py now works -> auto input_dims variable
- Small change to clip_model.py to accept this variable
- In clip.py, bypass SHA256 checksum verification -> You can put your fine-tune in place of .cache/clip/<original_model>.pt
- Include model.py from OpenAI/clip -> config for fine-tuned torch.save .pt files w/o inbuilt model config
- Save plots rather than using plt.show()
- ⚠️ Note: No changes made to demo.ipynb - use demo.py from the console!

----
## Advanced:

- Use runall.py (type "python runall.py --help"). Will:
- Batch process images + perform CLIP Surgery in a fully automated way:
- 1. Gets some CLIP opinions in gradient ascent -> model's own texts (labels) about the images.
- 2. Performs CLIP Surgery with whatever CLIP "saw" in the images.
- ⚠️ You can use large models, but from CLIP ViT-L/14 on, will require >24 GB memory. 
- FUN: After [above], run FUN_word-world-records.py to get a list of CLIP's craziest predicted longwords.
- ℹ️ Requires: Original OpenAI/CLIP "pip install git+https://github.com/openai/CLIP.git"

---
- Original CLIP Gradient Ascent Script: Used with permission by Twitter / X: [@advadnoun](https://twitter.com/advadnoun)
- CLIP 'opinions' may contain biased rants, slurs, and profanity. For more information, refer to the [CLIP Model Card](https://github.com/openai/CLIP/blob/main/model-card.md).

![example-github](https://github.com/zer0int/CLIP_Surgery/assets/132047210/e4b4f6ec-8dd5-46a9-8dac-8d5755ef70ea)

------------

# ORIGINAL README.md:
# CLIP Surgery for Better Explainability with Enhancement in Open-Vocabulary Tasks ([arxiv](https://arxiv.org/abs/2304.05653))

## Introduction

This work focuses on the explainability of CLIP via its raw predictions. We identify two problems about CLIP's explainability: opposite visualization and noisy activations. Then we propose the CLIP Surgery, which does not require any fine-tuning or additional supervision. It greatly improves the explainability of CLIP, and enhances downstream open-vocabulary tasks such as multi-label recognition, semantic segmentation, interactive segmentation (specifically the Segment Anything Model, SAM), and multimodal visualization. Currently, we offer a simple demo for interpretability analysis, and how to convert text to point prompts for SAM. Rest codes including evaluation and other tasks will be released later.

Opposite visualization is due to wrong relation in self-attention:
![image](figs/fig1.jpg)

Noisy activations is owing to redundant features across lables:
![image](figs/fig2.jpg)

Our visualization results:
![image](figs/fig3.jpg)

Text2Points to guide SAM:
![image](figs/fig4.jpg)

Multimodal visualization:
![image](figs/fig5.jpg)

Segmentation results:
![image](figs/fig6.jpg)

Multilabel results:
![image](figs/fig7.jpg)

## Demo

Firstly to install the SAM, and download the model
```
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Then explain CLIP via jupyter demo ["demo.ipynb"](https://github.com/xmed-lab/CLIP_Surgery/blob/master/demo.ipynb).
Or use the python file:
```
python demo.py
```
(Note: demo's results are slightly different from the experimental code, specifically no apex amp fp16 for easier to use.)

## Cite
```
@misc{li2023clip,
      title={CLIP Surgery for Better Explainability with Enhancement in Open-Vocabulary Tasks}, 
      author={Yi Li and Hualiang Wang and Yiqun Duan and Xiaomeng Li},
      year={2023},
      eprint={2304.05653},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
