# Divide, Evaluate, and Refine: Evaluating and Improving Text-to-Image Alignment with Iterative VQA Feedback (NeurIPS 2023)

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

<a href="https://arxiv.org/abs/2307.04749"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=for-the-badge" height=22.5></a>
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

Official Implementation for our NeurIPS-2023 paper on [Divide, Evaluate, and Refine: Evaluating and Improving Text-to-Image Alignment with Iterative VQA Feedback](https://1jsingh.github.io/divide-evaluate-and-refine). 

## Updates

* **(09/11/23)** Code for both evaluation and improvement of T2I generation is now available as a [diffusers](https://github.com/huggingface/diffusers) pipeline.

## Quick Links
  * [Getting Started](#getting-started)
    + [Prerequisites](#prerequisites)
    + [Setup](#setup)
  * [QuickStart Demo](#quickstart-demo-notebooks)
  * [DA-Score: Evaluating Text to Image Aligment](#da-score--evaluating-text-to-image-aligment)
  * [Divide-Evaluate and Refine: Improving Text to Image Aligment](#divide-evaluate-and-refine--improving-text-to-image-aligment)
  * [Citation](#citation)

## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
- Python 3
- Tested on Ubuntu 20.04, Nvidia RTX 3090 and CUDA 11.5 (though will likely run on other setups without modification)

### Setup
- Dependencies:  
To set up the environment, please run:
```
conda env create -f environment/environment.yml
conda activate dascore
```


## QuickStart Demo
We provide a quickstart demo notebook `demo.ipynb` to get started with Divide-Evaluate-and-Refine. The demo notebook includes a step-by-step analysis including:

* Using DA-Score for evaluating text to image alignment.
* Using Eval&Refine for gradually improving the quality of generated image outputs.
* Analysing three main ways in which Eval&Refine improves over Attend&Excite.

## DA-Score: Evaluating Text to Image Aligment

<!-- ### Usage -->
Evaluation of T2I alignment can be done in just few lines of code:

```python
import openai
from t2i_eval.utils import generate_questions, VQAModel, compute_dascores

openai.api_key = "[Your OpenAI Key]"

# create vqa_model
vqa_model = VQAModel()

# generate set of disjoint questions from the input prompt using LLM model
questions, parsed_input = generate_questions(prompt)

# compute DA-Score from the generated questions
da_score, assertion_alignment_scores = compute_dascores([image], questions, vqa_model)
```

<!-- ### Large-scale Evaluation -->

<!-- ## Decomposable-Captions-4k Dataset -->

<!-- ### Large-scale Evaluation -->

## Divide-Evaluate and Refine: Improving Text to Image Aligment
Assertion/Question level alignment scores also provide a useful feedback to determine which parts of the input prompt are not being expressed in the final image generation. Eval&Refine uses this knowledge in order to propose a very simple yet effective iterative refinement process which gradually improves the quality of the final images.

<!-- ### Usage -->

Eval&Refine is available as a convenient [diffusers](https://github.com/huggingface/diffusers) pipeline for easy use.

* Quick Usage:
``` python
from t2i_improve.pipeline_evaluate_and_refine import StableDiffusionEvaluatendRefinePipeline
from t2i_eval.utils import generate_questions, VQAModel, compute_dascores

# import openai api
import openai
openai.api_key = "[Your OpenAI Key]"

# define and load the Pipeline
pipe = StableDiffusionEvaluatendRefinePipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
# create vqa_model for DA-Score
vqa_model = VQAModel()

# define prompt and generate parsed_input using LLM model
prompt = 'a penguin wearing a bowtie on a surfboard in a swimming pool'
questions, parsed_input = generate_questions(prompt)

# perform inference using iterative refinement
outputs = pipe.eval_and_refine(parsed_input, vqa_model, seed_list=[77,402], max_update_steps=5, verbose=False)
```

Notes: 
 * We provide two mechanisms for iterative refinement with Eval&Refine: 
    - Prompt-Weighting (PW)
    - Cross-Attention modulation (CA).
    The use of the above can be controlled through `use_pw` and `use_ca` keywords while calling the pipeline. For e.g. to use Prompt-Weighting (PW) but not Cross-Attention modulation (CA), please use:
    ```python
    outputs = pipe.eval_and_refine(parsed_input, vqa_model, use_pw=True, use_ca=False,  max_update_steps=5)
    ```
  * The maximum number of iterative-refinement steps can be controlled by `max_update_steps` parameter. More iterative refinement can potentially help get better image outputs. Eval&Refine can adaptively adjust the actual number of refinement steps by monitoring the DA-Score.

  * The threshold for what is considered as a good enough output it controlled by the `dascore_threshold=0.85` and `assertion_alignment_threshold=0.75`. The iterative refinement process considers the final output image to be good enough if the overall DA-Score is greator the `dascore_threshold` or all individual alignment-scores for each assertion are greator then `assertion_alignment_threshold`

  * Reducing the above thresolds can lead to faster convergence at cost of poor image quality and vice versa.

 * Finally, we can visualize how the iterative refinement process gradually improves the generated image outputs by setting `verbose=True` while calling the pipeline.
 ```python
 outputs = pipe.eval_and_refine(parsed_input, vqa_model, max_update_steps=5, verbose=True)
 ```

## Citation

If you find our work useful in your research, please consider citing:
```
@inproceedings{singh2023divide,
  title={Divide, Evaluate, and Refine: Evaluating and Improving Text-to-Image Alignment with Iterative VQA Feedback},
  author={Singh, Jaskirat and Zheng, Liang},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```


