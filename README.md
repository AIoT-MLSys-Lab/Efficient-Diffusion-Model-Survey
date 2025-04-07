# Efficient Diffusion Models: A Survey
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align="center">

   [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
   [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FNastyMarcus%2FA-Survey-of-Efficient-Diffusion-Models&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
   ![GitHub Repo stars](https://img.shields.io/github/stars/AIoT-MLSys-Lab/Efficient-Diffusion-Model-Survey)

</h5>

Diffusion models have kickstart a new era in the field of artificial intelligence generative content (AIGC). This repo is a curated list of papers about the latest advancements in efficient diffusion models. **This repo is being actively updated, please stay tuned!**

## 📣 Update News

`[2024-11-9]` We released the repository.

## ❤️ Community Support

We will actively maintain this repository by incorporating new research as it emerges. If you have any suggestions regarding our taxonomy, find any missed papers, or update any preprint arXiv paper that has been accepted to some venue, feel free to send us an email or submit a **pull request** using the following markdown format.


## 📖 Table of Contents
<!-- TOC -->
- [Efficient Diffusion Models: A Survey](#efficient-diffusion-models-a-survey)
  - [📣 Update News](#-update-news)
  - [❤️ Community Support](#️-community-support)
  - [📖 Table of Contents](#-table-of-contents)
    - [Algorithm](#algorithm)
      - [Efficient Training](#efficient-training)
      - [Efficient Fine-tuning](#efficient-fine-tuning)
      - [Efficient Sampling](#efficient-sampling)
      - [Compression](#compression)
    - [System](#system)
      - [Hardware-Software Co-Design](#hardware-software-co-design)
      - [Parallel Computing](#parallel-computing)
      - [Caching Technique](#caching-technique)
    - [Frameworks](#frameworks)
  - [👍 Acknowledgement](#-acknowledgement)
  - [📑 Citation](#-citation)


### Algorithm
#### Efficient Training
  - ##### Noise Schedule
    - **[NeurIPS 2020]** Denoising Diffusion Probabilistic Models. [[Paper]](https://arxiv.org/pdf/2006.11239) [[Code]](https://github.com/hojonathanho/diffusion)
    - **[ICLR 2021]** Denoising Diffusion Implicit Models. [[Paper]](https://openreview.net/pdf?id=St1giarCHLP)
    - **[ICML 2021]** Improved Denoising Diffusion Probabilistic Models. [[Paper]](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf) [[Code]](https://github.com/openai/improved-diffusion)
    - **[Arxiv 2024.07]** Improved Noise Schedule for Diffusion Training. [[Paper]](https://arxiv.org/pdf/2407.03297) 
    - **[EMNLP 2023]** A Cheaper and Better Diffusion Language Model with Soft-Masked Noise. [[Paper]](https://aclanthology.org/2023.emnlp-main.289.pdf) [[Code]](https://github.com/SALT-NLP/Masked_Diffusioin_LM)
    - **[NeurIPS 2024]** ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/2ac2eac5098dba08208807b65c5851cc-Paper-Conference.pdf) [[Code]](https://github.com/zsyOAOA/ResShift)
    - **[Arxiv 2024.06]** Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment. [[Paper]](https://arxiv.org/pdf/2406.12303) [[Code]](https://github.com/yhli123/immiscible-diffusion)
    - **[ICLR 2023]** DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models. [[Paper]](https://arxiv.org/pdf/2210.08933) [[Code]](https://github.com/Shark-NLP/DiffuSeq)
    - **[ACL 2024]** Text Diffusion Model with Encoder-Decoder Transformers for Sequence-to-Sequence Generation. [[Paper]](https://aclanthology.org/2024.naacl-long.2.pdf)

  - ##### Score Matching
    - **[JMLR 2005]** Estimation of Non-Normalized Statistical Models by Score Matching. [[Paper]](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)
    - **[UAI 2019]** Sliced Score Matching: A Scalable Approach to Density and Score Estimation. [[Paper]](https://proceedings.mlr.press/v115/song20a/song20a.pdf) [[Code]](https://github.com/ermongroup/sliced_score_matching)
    - **[NeurIPS 2019]** Generative Modeling by Estimating Gradients of the Data Distribution. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf) [[Code]](https://github.com/kasyap1234/-Generative-Modeling-by-Estimating-Gradients-of-the-Data-Distribution-implementation)
    - **[NeurIPS 2021]** Maximum Likelihood Training of Score-Based Diffusion Models. [[Paper]](https://proceedings.neurips.cc/paper/2021/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf) [[Code]](https://github.com/yang-song/score_flow) 
    - **[ICLR 2022]** Score-Based Generative Modeling with Critically-Damped Langevin Diffusion. [[Paper]](https://arxiv.org/pdf/2112.07068) [[Code]](https://github.com/nv-tlabs/CLD-SGM)

  - ##### Data-Dependent Adaptive Priors
    - **[Arxiv 2022.02]** PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior. [[Paper]](https://arxiv.org/pdf/2106.06406)
    - **[Arxiv 2023.05]** DiGress: Discrete Denoising diffusion for graph generation. [[Paper]](https://arxiv.org/pdf/2209.14734) [[Code]](https://github.com/cvignac/DiGress)
    - **[Arxiv 2024.02]** DecompDiff: Diffusion Models with Decomposed Priors for Structure-Based Drug Design. [[Paper]](https://arxiv.org/pdf/2403.07902) [[Code]](https://github.com/bytedance/DecompDiff)
    - **[CVPR 2023]** Leapfrog diffusion model for stochastic trajectory prediction. [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Mao_Leapfrog_Diffusion_Model_for_Stochastic_Trajectory_Prediction_CVPR_2023_paper.pdf) [[Code]](https://github.com/MediaBrain-SJTU/LED)
  
  - ##### Rectified Flow
    - **[Arxiv 2022.09]** Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. [[Paper]](https://arxiv.org/pdf/2209.03003) [[Code]](https://github.com/gnobitab/RectifiedFlow)
    - **[ICLR 2024]** InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation. [[Paper]](https://openreview.net/pdf?id=1k4yZbbDqX) [[Code]](https://github.com/gnobitab/InstaFlow)
    - **[Arxiv 2022.09]** Rectified Flow: A Marginal Preserving Approach to Optimal Transport. [[Paper]](https://arxiv.org/pdf/2209.14577)
    - **[Arxiv 2024.10]** Improving the Training of Rectified Flows. [[Paper]](https://arxiv.org/pdf/2405.20320) [[Code]](https://github.com/sangyun884/rfpp)
    - **[Arxiv 2023.09]** Diffusion Models with Deterministic Normalizing Flow Priors. [[Paper]](https://arxiv.org/pdf/2309.01274) [[Code]](https://github.com/MohsenZand/DiNof)
    - **[Arxiv 2024.09]** PeRFlow: Piecewise Rectified Flow as Universal Plug-and-Play Accelerator. [[Paper]](https://arxiv.org/pdf/2405.07510) [[Code]](https://github.com/magic-research/piecewise-rectified-flow)
    - **[Arxiv 2024.02]** SlimFlow: Training Smaller One-Step Diffusion Models with Rectified Flow. [[Paper]](https://arxiv.org/pdf/2407.12718?) [[Code]](https://github.com/yuanzhi-zhu/SlimFlow)
  
#### Efficient Fine-tuning

  - ##### Low-Rank Adaptation
    - **[Arxiv 2024.10]** LoRA: Low-Rank Adaptation of Large Language Models. [[Paper]](https://arxiv.org/pdf/2106.09685) [[Code]](https://github.com/microsoft/LoRA)
    - **[ECCV 2024]** Lcm-lora: A universal stable-diffusion acceleration module [[Paper]](https://arxiv.org/pdf/2311.05556) [[Code]](https://github.com/luosiallen/latent-consistency-model)
    - **[ECCV 2024]** Concept sliders: Lora adaptors for precise control in diffusion models. [[Paper]](https://arxiv.org/pdf/2311.12092) [[Code]](https://github.com/rohitgandikota/sliders)
    - **[Arxiv 2024.07]** LoRA-Composer: Leveraging Low-Rank Adaptation for Multi-Concept Customization in Training-Free Diffusion Models. [[Paper]](https://arxiv.org/pdf/2403.11627)  [[Code]](https://github.com/Young98CN/LoRA_Composer)
    - **[Arxiv 2024.10]** Simple Drop-in LoRA Conditioning on Attention Layers Will Improve Your Diffusion Model [[Paper]](https://arxiv.org/pdf/2405.03958)

  - ##### Adapter
    - **[AAAI 2024]** T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28226/28447) [[Code]](https://github.com/TencentARC/T2I-Adapter.)
    - **[Arxiv 2023.08]** Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. [[Paper]](https://arxiv.org/pdf/2308.06721) [[Code]](https://github.com/tencent-ailab/IP-Adapter)
    - **[Arxiv 2024.05]** Ctrl-Adapter: An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model [[Paper]](https://arxiv.org/pdf/2404.09967) [[Code]](https://github.com/HL-hanlin/Ctrl-Adapter)
    - **[CVPR 2024]** Simda: Simple diffusion adapter for efficient video generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Xing_SimDA_Simple_Diffusion_Adapter_for_Efficient_Video_Generation_CVPR_2024_paper.pdf) [[Code]](https://chenhsing.github.io/SimDA/)

  - ##### ControlNet
    - **[ICCV 2023]** Adding conditional control to text-to-image diffusion models. [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-To-Image_Diffusion_Models_ICCV_2023_paper.pdf)[[Code]](https://github.com/faverogian/controlNet)
    - **[ECCV 2025]** ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback. [[Paper]](https://arxiv.org/pdf/2404.07987)[[Code]](https://github.com/liming-ai/ControlNet_Plus_Plus)
    - **[Arxiv 2023.12]** ControlNet-XS: Rethinking the Control of Text-to-Image Diffusion Models as Feedback-Control Systems. [[Paper]](https://arxiv.org/pdf/2312.06573)[[Code]](https://github.com/vislearn/ControlNet-XS)
    - **[Arxiv 2024.08]** ControlNext: Powerful and efficient control for image and video generation. [[Paper]](https://arxiv.org/pdf/2408.06070)[[Code]](https://github.com/dvlab-research/ControlNeXt)
    - **[NeurIPS 2024]** Uni-ControlNet: All-in-one control to text-to-image diffusion models. [[Paper]](https://arxiv.org/pdf/2305.16322)[[Code]](https://github.com/ShihaoZhaoZSH/Uni-ControlNet)
    - **[NeurIPS 2023]** UniControl: A unified diffusion model for controllable visual generation in the wild. [[Paper]](https://openreview.net/pdf?id=v54eUIayFh)[[Code]](https://github.com/salesforce/UniControl)


#### Efficient Sampling

  - ##### Solver
    - **[ICLR 2021]** Score-Based Generative Modeling through Stochastic Differential Equations. [[Paper]](https://arxiv.org/pdf/2011.13456) [[Code]](https://github.com/yang-song/score_sde)
    - **[NeurIPS 2021]** Diffusion Normalizing Flow. [[Paper]](https://proceedings.neurips.cc/paper/2021/file/876f1f9954de0aa402d91bb988d12cd4-Paper.pdf)
    - **[Arxiv 2021.05]** Gotta Go Fast When Generating Data with Score-Based Models. [[Paper]](https://arxiv.org/pdf/2105.14080) [[Code]](https://github.com/AlexiaJM/score_sde_fast_sampling)
    - **[NeurIPS 2023]** Gaussian Mixture Solvers for Diffusion Models. [[Paper]](https://papers.nips.cc/paper_files/paper/2023/file/51373b6499708b6fcc38f1e8f8f5b376-Paper-Conference.pdf) [[Code]](https://github.com/Guohanzhong/GMS)
    - **[ICML 2024]** Unifying Bayesian Flow Networks and Diffusion Models through Stochastic Differential Equations. [[Paper]](https://openreview.net/pdf/4d120b565267ca44bc866a8f372f670c5837e719.pdf) [[Code]](https://github.com/ML-GSAI/BFN-Solver)
    - **[NeurIPS 2023]** SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models. [[Paper]](https://openreview.net/pdf?id=f6a9XVFYIo)
    - **[NeurIPS 2022]** DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. [[Paper]](https://arxiv.org/pdf/2206.00927) [[Code]](https://github.com/LuChengTHU/dpm-solver)
    - **[ICLR 2023]** Fast Sampling of Diffusion Models with Exponential Integrator. [[Paper]](https://openreview.net/pdf?id=Loek7hfb46P) [[Code]](https://github.com/qsh-zh/deis)
    - **[ICML 2023]** Improved Techniques for Maximum Likelihood Estimation for Diffusion ODEs. [[Paper]](https://proceedings.mlr.press/v202/zheng23c/zheng23c.pdf) [[Code]](https://github.com/thu-ml/i-DODE)
    - **[ICML 2023]** Denoising MCMC for Accelerating Diffusion-Based Generative Models. [[Paper]](https://proceedings.mlr.press/v202/kim23z/kim23z.pdf) [[Code]](https://github.com/1202kbs/DMCMC)

  - ##### Sampling Scheduling
    - **[NeurIPS 2022]** Deep Equilibrium Approaches to Diffusion Models. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/f7f47a73d631c0410cbc2748a8015241-Paper-Conference.pdf) [[Code]](https://github.com/ashwinipokle/deq-ddim)
    - **[ICML 2024]** Accelerating Parallel Sampling of Diffusion Models. [[Paper]](https://openreview.net/pdf?id=CjVWen8aJL) [[Code]](https://github.com/TZW1998/ParaTAA-Diffusion)
    - **[NeurIPS 2023]** Parallel Sampling of Diffusion Models. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/0d1986a61e30e5fa408c81216a616e20-Paper-Conference.pdf) [[Code]](https://github.com/AndyShih12/paradigms)

  - ##### Truncated Sampling
    - **[EMNLP 2021]** Consistent Accelerated Inference via Confident Adaptive Transformers. [[Paper]](https://arxiv.org/pdf/2104.08803) [[Code]](https://github.com/TalSchuster/CATs)
    - **[NeurIPS 2022]** Confident Adaptive Language Modeling. [[Paper]](https://openreview.net/pdf?id=uLYc4L3C81A)
    - **[ICML 2024]** A Simple Early Exiting Framework for Accelerated Sampling in Diffusion Models. [[Paper]](https://openreview.net/pdf?id=OnOaj3g9fi) [[Code]](https://github.com/taehong-moon/ee-diffusion)
    - **[Arxiv 2022.04]** Semi-Parametric Neural Image Synthesis. [[Paper]](https://arxiv.org/pdf/2204.11824) [[Code]](https://github.com/lucidrains/retrieval-augmented-ddpm)
    - **[ICLR 2023]** kNN-Diffusion: Image Generation via Large-Scale Retrieval. [[Paper]](https://openreview.net/pdf?id=x5mtJD2ovc)
    - **[ICLR 2023]** Re-Imagen: Retrieval-Augmented Text-to-Image Generator. [[Paper]](https://openreview.net/pdf?id=XSEBx0iSjFQ)
    - **[ICML 2023]** ReDi: Efficient Learning-Free Diffusion Inference via Trajectory Retrieval. [[Paper]](https://openreview.net/pdf?id=SP01yVIC2o) [[Code]](https://github.com/zkx06111/ReDiffusion)

#### Compression

  - ##### Quantization
    - **[CVPR 2023]** Post-training Quantization on Diffusion Models. [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shang_Post-Training_Quantization_on_Diffusion_Models_CVPR_2023_paper.pdf) [[Code]](https://github.com/42Shawn/PTQ4DM)
    - **[ICCV 2023]** Q-Diffusion: Quantizing Diffusion Models. [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Q-Diffusion_Quantizing_Diffusion_Models_ICCV_2023_paper.pdf) [[Code]](https://github.com/Xiuyu-Li/q-diffusion)
    - **[ICLR 2021]** BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction. [[Paper]](https://arxiv.org/pdf/2102.05426) [[Code]](https://github.com/yhhhli/BRECQ)
    - **[NeurIPS 2023]** Leveraging Early-Stage Robustness in Diffusion Models for Efficient and High-Quality Image Synthesis. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/04261fce1705c4f02f062866717d592a-Paper-Conference.pdf) 
    - **[NeurIPS 2023]** PTQD: Accurate Post-Training Quantization for Diffusion Models. [[Paper]](https://arxiv.org/pdf/2305.10657) [[Code]](https://github.com/ziplab/PTQD)
    - **[NeurIPS 2023]** Temporal Dynamic Quantization for Diffusion Models. [[Paper]](https://arxiv.org/pdf/2306.02316)
    - **[NeurIPS 2024]** BitsFusion: 1.99 bits Weight Quantization of Diffusion Model. [[Paper]](https://arxiv.org/pdf/2406.04333)
    - **[ICLR 2020]** Learned Step Size Quantization. [[Paper]](https://openreview.net/pdf?id=rkgO66VKDS) 
    - **[ICLR 2024]** EfficientDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models. [[Paper]](https://arxiv.org/pdf/2310.03270) [[Code]](https://github.com/ThisisBillhe/EfficientDM)

  - ##### Pruning
    - **[NeurIPS 2023]** Structural Pruning for Diffusion Models. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/35c1d69d23bb5dd6b9abcd68be005d5c-Paper-Conference.pdf) [[Code]](https://github.com/VainF/Diff-Pruning)
    - **[CVPRW 2024]** LD-Pruner: Efficient Pruning of Latent Diffusion Models using Task-Agnostic Insights. [[Paper]](https://openaccess.thecvf.com/content/CVPR2024W/EDGE/papers/Castells_LD-Pruner_Efficient_Pruning_of_Latent_Diffusion_Models_using_Task-Agnostic_Insights_CVPRW_2024_paper.pdf)
    - **[ICML 2024]** LayerMerge: Neural Network Depth Compression through Layer Pruning and Merging. [[Paper]](https://openreview.net/pdf?id=uDoy7AGvEC) [[Code]](https://github.com/snu-mllab/LayerMerge)
    - **[Arxiv 2024.04]** LAPTOP-Diff: Layer Pruning and Normalized Distillation for Compressing Diffusion Models. [[Paper]](https://arxiv.org/pdf/2404.11098) 

  - ##### Knowledge Distillation
    - **[Arxiv 2021.01]** Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed. [[Paper]](https://arxiv.org/pdf/2101.02388) [[Code]](https://github.com/tcl9876/Denoising_Student)
    - **[ICLR 2022]** Progressive Distillation for Fast Sampling of Diffusion Models. [[Paper]](https://openreview.net/pdf?id=TIdIXIpzhoI) [[Code]](https://github.com/google-research/google-research/tree/master/diffusion_distillation)
    - **[CVPR 2023]** On Distillation of Guided Diffusion Models. [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Meng_On_Distillation_of_Guided_Diffusion_Models_CVPR_2023_paper.pdf)
    - **[ICML 2023]** Consistency Models. [[Paper]](https://arxiv.org/pdf/2303.01469) [[Code]](https://github.com/openai/consistency_models)
    - **[ECCV 2020]** NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. [[Paper]](https://arxiv.org/pdf/2003.08934) [[Code]](https://github.com/bmild/nerf)
    - **[ICLR 2023]** DreamFusion: Text-to-3D using 2D Diffusion. [[Paper]](https://openreview.net/pdf?id=FjNys5c7VyY) [[Project]](https://dreamfusion3d.github.io/)
    - **[NeurIPS 2023]** ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation. [[Paper]](https://arxiv.org/pdf/2305.16213) [[Code]](https://github.com/thu-ml/prolificdreamer)
    - **[NeurIPS 2023]** Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models. [[Paper]](https://arxiv.org/pdf/2305.18455) [[Code]](https://github.com/pkulwj1994/diff_instruct)
    - **[CVPR 2024]** 3D Paintbrush: Local Stylization of 3D Shapes with Cascaded Score Distillation. [[Paper]](https://arxiv.org/pdf/2311.09571) [[Code]](https://github.com/threedle/3d-paintbrush)
    - **[CVPR 2025]** MoFlow: One-Step Flow Matching for Human Trajectory Forecasting via Implicit Maximum Likelihood Estimation based Distillation [[Paper]](https://arxiv.org/pdf/2503.09950) [[Code]](https://github.com/felix-yuxiang/MoFlow)

### System

#### Hardware-Software Co-Design
  - **[CVPRW 2023]** Speed Is All You Need: On-Device Acceleration of Large Diffusion Models via GPU-Aware Optimizations. [[Paper]](https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Chen_Speed_Is_All_You_Need_On-Device_Acceleration_of_Large_Diffusion_CVPRW_2023_paper.pdf) [[Project]](https://research.google/blog/speed-is-all-you-need-on-device-acceleration-of-large-diffusion-models-via-gpu-aware-optimizations/)
  - **[FPL 2024]** SDA: Low-Bit Stable Diffusion Acceleration on Edge FPGAs. [[Paper]](https://www.sfu.ca/~zhenman/files/C41-FPL2024-SDA.pdf) [[Code]](https://github.com/Michaela1224/SDA_code)
  - **[ISCAS 2024]** A 28.6 mJ/iter Stable Diffusion Processor for Text-to-Image Generation with Patch Similarity-based Sparsity Augmentation and Text-based Mixed-Precision. [[Paper]](https://arxiv.org/pdf/2403.04982)

#### Parallel Computing
  - **[CVPR 2024]** DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models. [[Paper]](https://arxiv.org/pdf/2402.19481) [[Code]](https://github.com/mit-han-lab/distrifuser)
  - **[Arxiv 2024.05]** PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models. [[Paper]](https://arxiv.org/pdf/2405.14430) [[Code]](https://github.com/PipeFusion/PipeFusion?tab=readme-ov-file)
  - **[Arxiv 2024.07]** SwiftDiffusion: Efficient Diffusion Model Serving with Add-on Modules. [[Paper]](https://arxiv.org/pdf/2407.02031) 
  - **[MLSys 2024]** DiffusionPipe: Training Large Diffusion Models with Efficient Pipelines. [[Paper]](https://proceedings.mlsys.org/paper_files/paper/2024/file/45c1f6a8cbf2da59ebf2c802b4f742cd-Paper-Conference.pdf)

#### Caching Technique
  - **[NSDI 2024]** Approximate Caching for Efficiently Serving Text-to-Image Diffusion Models. [[Paper]](https://arxiv.org/pdf/2312.04429) [[Code]](https://github.com/mit-han-lab/distrifuser)
  - **[CVPR 2024]** DeepCache: Accelerating Diffusion Models for Free. [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_DeepCache_Accelerating_Diffusion_Models_for_Free_CVPR_2024_paper.pdf) [[Code]](https://github.com/horseee/DeepCache)
  - **[CVPR 2024]** Cache Me if You Can: Accelerating Diffusion Models through Block Caching. [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Wimbauer_Cache_Me_if_You_Can_Accelerating_Diffusion_Models_through_Block_CVPR_2024_paper.pdf) [[Project]](https://fwmb.github.io/blockcaching/#)
  - **[Arxiv 2024.06]** Learning-to-Cache: Accelerating Diffusion Transformer via Layer Caching. [[Paper]](https://arxiv.org/pdf/2406.01733) [[Code]](https://github.com/horseee/learning-to-cache)
  - **[Arxiv 2024.07]** FORA: Fast-Forward Caching in Diffusion Transformer Acceleration. [[Paper]](https://arxiv.org/pdf/2407.01425) [[Code]](https://github.com/prathebaselva/FORA)


### Frameworks
<div align="center">

|                                                                               | Efficient Training | Efficient Fine-Tuning | Efficient Inference |
| :---------------------------------------------------------------------------- | :----------------: | :-------------------: | :-----------------: |
| Diffusers [[Code](https://github.com/huggingface/diffusers)]                  |         ✅          |           ✅           |          ✅          |
| DALL-E [[Code](https://github.com/openai/DALL-E)]                             |         ❌          |           ❌           |          ✅          |
| OneDiff [[Code](https://github.com/siliconflow/onediff)]                      |         ❌          |           ❌           |          ✅          |
| LiteGen [[Code](https://github.com/Vchitect/LiteGen)]                         |         ✅          |           ✅           |          ✅          |
| InvokeAI [[Code](https://github.com/invoke-ai/InvokeAI)]                      |         ❌          |           ✅           |          ✅          |
| ComfyUI-Docker [[Code](https://github.com/YanWenKun/ComfyUI-Docker)]          |         ❌          |           ✅           |          ✅          |
| Grate [[Code](https://github.com/damian0815/grate)]                           |         ❌          |           ✅           |          ✅          |
| Versatile Diffusion [[Code](https://github.com/SHI-Labs/Versatile-Diffusion)] |         ✅          |           ✅           |          ✅          |
| UniDiffuser [[Code](https://github.com/thu-ml/unidiffuser)]                   |         ✅          |           ✅           |          ✅          |

</div>

<!-- ## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NastyMarcus/A-Survey-of-Efficient-Diffusion-Models&type=Date)](https://star-history.com/#NastyMarcus/A-Survey-of-Efficient-Diffusion-Models&Date)

## ♥️ Contributors

<a href="https://github.com/NastyMarcus/A-Survey-of-Efficient-Diffusion-Models/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=NastyMarcus/A-Survey-of-Efficient-Diffusion-Models" />
</a> -->


<!--
## 👍 Acknowledgement
To be continued


## 📑 Citation

Please consider citing 📑 our papers if our repository is helpful to your work, thanks sincerely!

```BibTeX
To be continued
–->
