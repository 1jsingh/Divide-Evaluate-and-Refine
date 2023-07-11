# Divide, Evaluate, and Refine: Evaluating and Improving Text-to-Image Alignment with Iterative VQA Feedback

> The field of text-conditioned image generation has made unparalleled progress with
the recent advent of latent diffusion models. While remarkable, as the complexity
of given text input increases, the state-of-the-art diffusion models may still fail
in generating images which accurately convey the semantics of the given prompt.
Furthermore, it has been observed that such misalignments are often left undetected
by pretrained multi-modal models such as CLIP. To address these problems, in
this paper we explore a simple yet effective decompositional approach towards
both evaluation and improvement of text-to-image alignment. In particular, we
first introduce a Decompositional-Alignment-Score which given a complex prompt
decomposes it into a set of disjoint assertions. The alignment of each assertion with
generated images is then measured using a VQA model. Finally, alignment scores
for different assertions are combined aposteriori to give the final text-to-image
alignment score. Experimental analysis reveals that the proposed alignment metric
shows significantly higher correlation with human ratings as opposed to traditional
CLIP, BLIP scores. Furthermore, we also find that the assertion level alignment
scores provide a useful feedback which can then be used in a simple iterative
procedure to gradually increase the expression of different assertions in the final
image outputs. Human user studies indicate that the proposed approach surpasses
previous state-of-the-art by 8.7% in overall text-to-image alignment accuracy.

<a href="https://1jsingh.github.io/publications/divide-evaluate-and-refine.pdf"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=for-the-badge" height=22.5></a>
<a href="https://1jsingh.github.io/divide-evaluate-and-refine"><img src="https://img.shields.io/badge/Project-Page-succees?style=for-the-badge&logo=GitHub" height=22.5></a>
<a href="#"><img src="https://img.shields.io/badge/Online-Demo-blue?style=for-the-badge&logo=Streamlit" height=22.5></a>
<a href="#citation"><img src="https://img.shields.io/badge/Paper-Citation-green?style=for-the-badge&logo=Google%20Scholar" height=22.5></a>
<!-- <a href="https://twitter.com/intent/tweet?url=https%3A%2F%2Fgithub.com%2F1jsingh%2Fpaint2pix&text=Unleash%20your%20inner%20artist%20...%20synthesize%20amazing%20artwork%2C%20and%20realistic%20image%20content%20or%20simply%20perform%20a%20range%20of%20diverse%20real%20image%20edits%20using%20just%20coarse%20user%20scribbles.&hashtags=Paint2Pix%2CECCV2022"><img src="https://img.shields.io/badge/Share--white?style=for-the-badge&logo=Twitter" height=22.5></a> -->

<p align="center">
<img src="./docs/overview-eval-v3.png" width="800px"/>  
<img src="./docs/overview-iter-v1.png" width="800px"/>  
<br>
We propose a training-free decompositional framework which helps both better evaluate (top) and gradually improve (bottom) text-to-image alignment using iterative VQA feedback.
</p>

## Description   

Official Implementation for our paper [Divide, Evaluate, and Refine: Evaluating and Improving Text-to-Image Alignment with Iterative VQA Feedback](https://1jsingh.github.io/divide-evaluate-and-refine). 

## Updates (Coming Soon)

* **(10/07/23)** The repo is under construction. Code and dataset will be released soon.

## Citation

If you use this code for your research, please consider citing:
```
@article{singh2023divide,
      title={Divide, Evaluate, and Refine: Evaluating and Improving Text-to-Image Alignment with Iterative VQA Feedback},
      author={Singh, Jaskirat and Zheng, Liang},
      journal={arXiv preprint},
      year={2023}
    }
```


