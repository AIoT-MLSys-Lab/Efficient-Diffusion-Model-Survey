# Efficient Diffusion Models: A Survey
> **[Efficient Diffusion Models: A Survey](https://arxiv.org/abs/2502.06805)**[ [arXiv]](https://arxiv.org/abs/2502.06805) (Version 1: 02/03/2025; Version 3: 06/06/2025, camera ready version of Transactions on Machine Learning Research (TMLR))

> *Hui Shen<sup>1</sup>, Jingxuan Zhang<sup>2</sup>, Boning Xiong<sup>3</sup>, Rui Hu<sup>4</sup>,Shoufa Chen<sup>1</sup>, Zhongwei Wan<sup>1</sup>, Xin Wang<sup>1</sup>, Yu Zhang<sup>5</sup>, Zixuan Gong<sup>5</sup>, Guangyin Bao<sup>5</sup>, Chaofan Tao<sup>6</sup>, Yongfeng Huang<sup>7</sup>, Ye Yuan<sup>8</sup>, Mi Zhang.<sup>1</sup>*

> *<sup>1</sup>The Ohio State University, <sup>2</sup>Indiana University, <sup>3</sup>Fudan University, <sup>4</sup>Hangzhou City University, <sup>5</sup>Tongji University, <sup>6</sup>The University of Hong Kong, <sup>7</sup>The Chinese University of Hong Kong, <sup>8</sup>Peking University.*

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align="center">

   [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
   ![GitHub Repo stars](https://img.shields.io/github/stars/AIoT-MLSys-Lab/Efficient-Diffusion-Model-Survey)

</h5>

## ⚡News: Our survey has been officially accepted by Transactions on Machine Learning Research (TMLR), May 2025.
```
@article{shen2025efficient,
  title={Efficient Diffusion Models: A Survey},
  author={Shen, Hui and Zhang, Jingxuan and Xiong, Boning and Hu, Rui and Chen, Shoufa and Wan, Zhongwei and Wang, Xin and Zhang, Yu and Gong, Zixuan and Bao, Guangyin and others},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2025}
}
```

## ❤️ Community Support

We will actively maintain this repository by incorporating new research as it emerges. If you have any suggestions regarding our taxonomy, find any missed papers, or update any preprint arXiv paper that has been accepted to some venue, feel free to send us an email or submit a **pull request** using the following markdown format.

```markdown
Paper Title, <ins>Conference/Journal/Preprint, Year</ins>  [[pdf](link)] [[other resources](link)].
```
## 📌 What is This Survey About?

Diffusion models have emerged as powerful generative models capable of producing highquality contents such as images, videos, audio, and text, demonstrating their potential to revolutionize digital content generation. However, these capabilities come at the cost of their significant resource demands and lengthy generation time, underscoring the need to develop efficient techniques for practical deployment. In this survey, we provide a systematic and comprehensive review of research on efficient diffusion models. We organize the literature in a taxonomy consisting of three main categories, covering distinct yet interconnected efficient diffusion model topics from <b>algorithm</b>, <b>system</b>, and <b>framework</b> perspective, respectively. We hope our survey can serve as a valuable resource to help researchers and practitioners gain a systematic understanding of efficient diffusion model research and inspire them to contribute to this important and exciting field.

## 📖 Table of Contents
- [🔢 Algorithm-Level Optimization](#-algorithm-level-optimization)
   - [Efficient Training](#efficient-training)
      - [Latent Diffusion](#latent-diffusion)
      - [Loss Formulation](#loss-formulation)
      - [Training Tricks](#training-tricks)
   - [Efficient Fine-tuning](#efficient-fine-tuning)
      - [Low-Rank Adaptation](#low-rank-adaptation)
      - [Adapter](#adapter)
      - [ControlNet](#controlnet)
   - [Efficient Sampling](#efficient-sampling)
      - [Solver](#solver)
      - [Sampling Scheduling](#sampling-scheduling)
      - [Truncated Sampling](#truncated-sampling)
      - [Knowledge Distillation](#knowledge-distillation)
   - [Compression](#compression)
      - [Quantization](#quantization)
      - [Pruning](#pruning)
- [🧑‍💻 System-Level Optimization](#-system-level-optimization)
   - [Hardware-Software Co-Design](#hardware-software-co-design)
   - [Parallel Computing](#parallel-computing)
   - [Caching Technique](#caching-technique)
- [🔧 Frameworks](#-frameworks)

## 🔢 Algorithm-Level Optimization
### Efficient Training
#### Latent Diffusion
- **[ICLR 2024]** Latent 3D Graph Diffusion. [[Paper]](https://openreview.net/pdf?id=cXbnGtO0NZ) [[Code]](https://github.com/Shen-Lab/LDM-3DG)
- **[Arxiv 2024.10]** L3DG: Latent 3D Gaussian Diffusion. [[Paper]](https://arxiv.org/pdf/2410.13530)
- **[Arxiv 2024.09]** Latent Diffusion Models for Controllable RNA Sequence Generation. [[Paper]](https://arxiv.org/pdf/2409.09828)
- **[ICLR 2024]** MIXED-TYPE TABULAR DATA SYNTHESIS WITH SCORE-BASED DIFFUSION IN LATENT SPACE [[Paper]](https://openreview.net/pdf?id=4Ay23yeuz0) [[Code]](https://github.com/amazon-science/tabsyn)
- **[CVPR 2023]** Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models. [[Paper]](https://arxiv.org/pdf/2304.08818)
- **[CVPR 2023]** Video Probabilistic Diffusion Models in Projected Latent Space. [[Paper]](https://arxiv.org/pdf/2302.07685)
- **[Arxiv 2023.03]** Latent Video Diffusion Models for High-Fidelity Long Video Generation. [[Paper]](https://arxiv.org/abs/2211.13221)
- **[CVPR 2023]** Executing your commands via motion diffusion in latent space. [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Executing_Your_Commands_via_Motion_Diffusion_in_Latent_Space_CVPR_2023_paper.pdf) [[Code]](https://github.com/chenfengye/motion-latent-diffusion)
- **[ICML 2023]** AudioLDM: Text-to-Audio Generation with Latent Diffusion Models. [[Paper]](https://arxiv.org/pdf/2301.12503) [[Code]](https://github.com/haoheliu/AudioLDM)
- **[NeurIPS 2023]** Generating behaviorally diverse policies with latent diffusion models. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/180d4373aca26bd86bf45fc50d1a709f-Paper-Conference.pdf)
- **[Arxiv 2022.11]** MagicVideo: Efficient Video Generation With Latent Diffusion Models. [[Paper]](https://arxiv.org/pdf/2211.11018)

#### Loss Formulation
- **[ICLR 2024]** InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation. [[Paper]](https://openreview.net/pdf?id=1k4yZbbDqX) [[Code]](https://github.com/gnobitab/InstaFlow)
- **[Arxiv 2024.09]** PeRFlow: Piecewise Rectified Flow as Universal Plug-and-Play Accelerator. [[Paper]](https://arxiv.org/pdf/2405.07510) [[Code]](https://github.com/magic-research/piecewise-rectified-flow)
- **[Arxiv 2024.10]** Improving the Training of Rectified Flows. [[Paper]](https://arxiv.org/pdf/2405.20320) [[Code]](https://github.com/sangyun884/rfpp)
- **[Arxiv 2024.02]** SlimFlow: Training Smaller One-Step Diffusion Models with Rectified Flow. [[Paper]](https://arxiv.org/pdf/2407.12718?) [[Code]](https://github.com/yuanzhi-zhu/SlimFlow)
- **[ICLR 2022]** Score-Based Generative Modeling with Critically-Damped Langevin Diffusion. [[Paper]](https://arxiv.org/pdf/2112.07068) [[Code]](https://github.com/nv-tlabs/CLD-SGM)
- **[Arxiv 2022.09]** Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. [[Paper]](https://arxiv.org/pdf/2209.03003) [[Code]](https://github.com/gnobitab/RectifiedFlow)
- **[Arxiv 2022.09]** Rectified Flow: A Marginal Preserving Approach to Optimal Transport. [[Paper]](https://arxiv.org/pdf/2209.14577)
- **[NeurIPS 2021]** Maximum Likelihood Training of Score-Based Diffusion Models. [[Paper]](https://proceedings.neurips.cc/paper/2021/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf) [[Code]](https://github.com/yang-song/score_flow) 
- **[UAI 2019]** Sliced Score Matching: A Scalable Approach to Density and Score Estimation. [[Paper]](https://proceedings.mlr.press/v115/song20a/song20a.pdf) [[Code]](https://github.com/ermongroup/sliced_score_matching)
- **[NeurIPS 2019]** Generative Modeling by Estimating Gradients of the Data Distribution. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf) [[Code]](https://github.com/kasyap1234/-Generative-Modeling-by-Estimating-Gradients-of-the-Data-Distribution-implementation)
- **[JMLR 2005]** Estimation of Non-Normalized Statistical Models by Score Matching. [[Paper]](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)

#### Training Tricks
- **[ICLR 2025]** Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think. [[Paper]](https://openreview.net/pdf?id=DJSZGGZYVi) [[Code]](https://github.com/sihyun-yu/REPA)
- **[Arxiv 2025.01]** Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models. [[Paper]](https://arxiv.org/pdf/2501.01423) [[Code]](https://github.com/hustvl/LightningDiT)
- **[Arxiv 2024.07]** Improved Noise Schedule for Diffusion Training. [[Paper]](https://arxiv.org/pdf/2407.03297) 
- **[NeurIPS 2024]** ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/2ac2eac5098dba08208807b65c5851cc-Paper-Conference.pdf) [[Code]](https://github.com/zsyOAOA/ResShift)
- **[Arxiv 2024.06]** Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment. [[Paper]](https://arxiv.org/pdf/2406.12303) [[Code]](https://github.com/yhli123/immiscible-diffusion)
- **[Arxiv 2024.02]** DecompDiff: Diffusion Models with Decomposed Priors for Structure-Based Drug Design. [[Paper]](https://arxiv.org/pdf/2403.07902) [[Code]](https://github.com/bytedance/DecompDiff)
- **[ACL 2024]** Text Diffusion Model with Encoder-Decoder Transformers for Sequence-to-Sequence Generation. [[Paper]](https://aclanthology.org/2024.naacl-long.2.pdf)
- **[Arxiv 2023.05]** DiGress: Discrete Denoising diffusion for graph generation. [[Paper]](https://arxiv.org/pdf/2209.14734) [[Code]](https://github.com/cvignac/DiGress)
- **[CVPR 2023]** Leapfrog diffusion model for stochastic trajectory prediction. [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Mao_Leapfrog_Diffusion_Model_for_Stochastic_Trajectory_Prediction_CVPR_2023_paper.pdf) [[Code]](https://github.com/MediaBrain-SJTU/LED)
- **[ICLR 2023]** DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models. [[Paper]](https://arxiv.org/pdf/2210.08933) [[Code]](https://github.com/Shark-NLP/DiffuSeq)
- **[EMNLP 2023]** A Cheaper and Better Diffusion Language Model with Soft-Masked Noise. [[Paper]](https://aclanthology.org/2023.emnlp-main.289.pdf) [[Code]](https://github.com/SALT-NLP/Masked_Diffusioin_LM)
- **[Arxiv 2022.02]** PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior. [[Paper]](https://arxiv.org/pdf/2106.06406)
- **[ICLR 2021]** Denoising Diffusion Implicit Models. [[Paper]](https://openreview.net/pdf?id=St1giarCHLP)
- **[ICML 2021]** Improved Denoising Diffusion Probabilistic Models. [[Paper]](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf) [[Code]](https://github.com/openai/improved-diffusion)
- **[NeurIPS 2020]** Denoising Diffusion Probabilistic Models. [[Paper]](https://arxiv.org/pdf/2006.11239) [[Code]](https://github.com/hojonathanho/diffusion)

### Efficient Fine-tuning
#### Low-Rank Adaptation
- **[Arxiv 2024.10]** LoRA: Low-Rank Adaptation of Large Language Models. [[Paper]](https://arxiv.org/pdf/2106.09685) [[Code]](https://github.com/microsoft/LoRA)
- **[ECCV 2024]** Concept sliders: Lora adaptors for precise control in diffusion models. [[Paper]](https://arxiv.org/pdf/2311.12092) [[Code]](https://github.com/rohitgandikota/sliders)
- **[ECCV 2024]** Lcm-lora: A universal stable-diffusion acceleration module [[Paper]](https://arxiv.org/pdf/2311.05556) [[Code]](https://github.com/luosiallen/latent-consistency-model)
- **[Arxiv 2024.07]** LoRA-Composer: Leveraging Low-Rank Adaptation for Multi-Concept Customization in Training-Free Diffusion Models. [[Paper]](https://arxiv.org/pdf/2403.11627)  [[Code]](https://github.com/Young98CN/LoRA_Composer)
- **[Arxiv 2024.10]** Simple Drop-in LoRA Conditioning on Attention Layers Will Improve Your Diffusion Model [[Paper]](https://arxiv.org/pdf/2405.03958)

#### Adapter
- **[AAAI 2024]** T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28226/28447) [[Code]](https://github.com/TencentARC/T2I-Adapter.)
- **[ICML 2024]** Accelerating Parallel Sampling of Diffusion Models. [[Paper]](https://openreview.net/pdf?id=CjVWen8aJL) [[Code]](https://github.com/TZW1998/ParaTAA-Diffusion)
- **[Arxiv 2024.05]** Ctrl-Adapter: An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model [[Paper]](https://arxiv.org/pdf/2404.09967) [[Code]](https://github.com/HL-hanlin/Ctrl-Adapter)
- **[CVPR 2024]** Simda: Simple diffusion adapter for efficient video generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Xing_SimDA_Simple_Diffusion_Adapter_for_Efficient_Video_Generation_CVPR_2024_paper.pdf) [[Code]](https://chenhsing.github.io/SimDA/)
- **[Arxiv 2023.08]** Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. [[Paper]](https://arxiv.org/pdf/2308.06721) [[Code]](https://github.com/tencent-ailab/IP-Adapter)

#### ControlNet
- **[ECCV 2025]** ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback. [[Paper]](https://arxiv.org/pdf/2404.07987)[[Code]](https://github.com/liming-ai/ControlNet_Plus_Plus)
- **[Arxiv 2024.08]** ControlNext: Powerful and efficient control for image and video generation. [[Paper]](https://arxiv.org/pdf/2408.06070)[[Code]](https://github.com/dvlab-research/ControlNeXt)
- **[NeurIPS 2024]** Uni-ControlNet: All-in-one control to text-to-image diffusion models. [[Paper]](https://arxiv.org/pdf/2305.16322)[[Code]](https://github.com/ShihaoZhaoZSH/Uni-ControlNet)
- **[NeurIPS 2023]** UniControl: A unified diffusion model for controllable visual generation in the wild. [[Paper]](https://openreview.net/pdf?id=v54eUIayFh)[[Code]](https://github.com/salesforce/UniControl)
- **[Arxiv 2023.12]** ControlNet-XS: Rethinking the Control of Text-to-Image Diffusion Models as Feedback-Control Systems. [[Paper]](https://arxiv.org/pdf/2312.06573)[[Code]](https://github.com/vislearn/ControlNet-XS)
- **[ICCV 2023]** Adding conditional control to text-to-image diffusion models. [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-To-Image_Diffusion_Models_ICCV_2023_paper.pdf)[[Code]](https://github.com/faverogian/controlNet)

### Efficient Sampling
#### Solver
- **[ICML 2024]** Unifying Bayesian Flow Networks and Diffusion Models through Stochastic Differential Equations. [[Paper]](https://openreview.net/pdf/4d120b565267ca44bc866a8f372f670c5837e719.pdf) [[Code]](https://github.com/ML-GSAI/BFN-Solver)
- **[NeurIPS 2023]** Gaussian Mixture Solvers for Diffusion Models. [[Paper]](https://papers.nips.cc/paper_files/paper/2023/file/51373b6499708b6fcc38f1e8f8f5b376-Paper-Conference.pdf) [[Code]](https://github.com/Guohanzhong/GMS)
- **[NeurIPS 2023]** SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models. [[Paper]](https://openreview.net/pdf?id=f6a9XVFYIo)
- **[NeurIPS 2023]** Restart sampling for improving generative processes. [[Paper]](https://openreview.net/pdf?id=wFuemocyHZ) [[Code]](https://github.com/Newbeeer/diffusion_restart_sampling)
- **[ICLR 2023]** Fast Sampling of Diffusion Models with Exponential Integrator. [[Paper]](https://openreview.net/pdf?id=Loek7hfb46P) [[Code]](https://github.com/qsh-zh/deis)
- **[ICML 2023]** Denoising MCMC for Accelerating Diffusion-Based Generative Models. [[Paper]](https://proceedings.mlr.press/v202/kim23z/kim23z.pdf) [[Code]](https://github.com/1202kbs/DMCMC)
- **[ICML 2023]** Improved Techniques for Maximum Likelihood Estimation for Diffusion ODEs. [[Paper]](https://proceedings.mlr.press/v202/zheng23c/zheng23c.pdf) [[Code]](https://github.com/thu-ml/i-DODE)
- **[Arxiv 2023.09]** Diffusion models with deterministic normalizing flow priors. [[Paper]](https://arxiv.org/pdf/2309.01274) [[Code]](https://github.com/MohsenZand/DiNof)
- **[NeurIPS 2022]** DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. [[Paper]](https://arxiv.org/pdf/2206.00927) [[Code]](https://github.com/LuChengTHU/dpm-solver)
- **[ICLR 2021]** Denoising diffusion implicit models. [[Paper]](https://openreview.net/pdf?id=St1giarCHLP)
- **[Arxiv 2021.05]** Gotta Go Fast When Generating Data with Score-Based Models. [[Paper]](https://arxiv.org/pdf/2105.14080) [[Code]](https://github.com/AlexiaJM/score_sde_fast_sampling)
- **[NeurIPS 2021]** Diffusion Normalizing Flow. [[Paper]](https://proceedings.neurips.cc/paper/2021/file/876f1f9954de0aa402d91bb988d12cd4-Paper.pdf)

#### Sampling Scheduling
- **[ICML 2024]** Align Your Steps: Optimizing Sampling Schedules in Diffusion Models. [[Paper]](https://openreview.net/pdf?id=nBGBzV4It3)
- **[ICML 2024]** Accelerating Parallel Sampling of Diffusion Models. [[Paper]](https://openreview.net/pdf?id=CjVWen8aJL) [[Code]](https://github.com/TZW1998/ParaTAA-Diffusion)
- **[NeurIPS 2023]** Parallel Sampling of Diffusion Models. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/0d1986a61e30e5fa408c81216a616e20-Paper-Conference.pdf) [[Code]](https://github.com/AndyShih12/paradigms)
- **[Arxiv 2023.12]** StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation. [[Paper]](https://arxiv.org/pdf/2312.12491) [[Code]](https://github.com/cumulo-autumn/StreamDiffusion)
- **[NeurIPS 2022]** Deep Equilibrium Approaches to Diffusion Models. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/f7f47a73d631c0410cbc2748a8015241-Paper-Conference.pdf) [[Code]](https://github.com/ashwinipokle/deq-ddim)
- **[Arxiv 2021.06]** On fast sampling of diffusion probabilistic models. [[Paper]](https://arxiv.org/pdf/2106.00132) [[Code]](https://github.com/FengNiMa/FastDPM_pytorch)
- **[Arxiv 2021.06]** Learning to Efficiently Sample from Diffusion Probabilistic Models. [[Paper]](https://arxiv.org/pdf/2106.03802)

   
#### Truncated Sampling
- **[ICML 2024]** A Simple Early Exiting Framework for Accelerated Sampling in Diffusion Models. [[Paper]](https://openreview.net/pdf?id=OnOaj3g9fi) [[Code]](https://github.com/taehong-moon/ee-diffusion)
- **[ICLR 2023]** kNN-Diffusion: Image Generation via Large-Scale Retrieval. [[Paper]](https://openreview.net/pdf?id=x5mtJD2ovc)
- **[ICLR 2023]** Re-Imagen: Retrieval-Augmented Text-to-Image Generator. [[Paper]](https://openreview.net/pdf?id=XSEBx0iSjFQ)
- **[ICML 2023]** ReDi: Efficient Learning-Free Diffusion Inference via Trajectory Retrieval. [[Paper]](https://openreview.net/pdf?id=SP01yVIC2o) [[Code]](https://github.com/zkx06111/ReDiffusion)
- **[Arxiv 2022.04]** Semi-Parametric Neural Image Synthesis. [[Paper]](https://arxiv.org/pdf/2204.11824) [[Code]](https://github.com/lucidrains/retrieval-augmented-ddpm)
- **[EMNLP 2021]** Consistent Accelerated Inference via Confident Adaptive Transformers. [[Paper]](https://arxiv.org/pdf/2104.08803) [[Code]](https://github.com/TalSchuster/CATs)

#### Knowledge Distillation
- **[CVPR 2025]** MoFlow: One-Step Flow Matching for Human Trajectory Forecasting via Implicit Maximum Likelihood Estimation based Distillation [[Paper]](https://arxiv.org/pdf/2503.09950) [[Code]](https://github.com/felix-yuxiang/MoFlow)
- **[Arxiv 2025.03]** SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation. [[Paper]](https://arxiv.org/pdf/2503.09641) [[Code]](https://github.com/NVlabs/Sana)
- **[CVPR 2024]** 3D Paintbrush: Local Stylization of 3D Shapes with Cascaded Score Distillation. [[Paper]](https://arxiv.org/pdf/2311.09571) [[Code]](https://github.com/threedle/3d-paintbrush)
- **[CVPR 2024]** One-step Diffusion with Distribution Matching Distillation. [[Paper]](https://arxiv.org/pdf/2311.18828)
- **[CVPR 2023]** On Distillation of Guided Diffusion Models. [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Meng_On_Distillation_of_Guided_Diffusion_Models_CVPR_2023_paper.pdf)
- **[ICLR 2023]** DreamFusion: Text-to-3D using 2D Diffusion. [[Paper]](https://openreview.net/pdf?id=FjNys5c7VyY) [[Project]](https://dreamfusion3d.github.io/)
- **[NeurIPS 2023]** ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation. [[Paper]](https://arxiv.org/pdf/2305.16213) [[Code]](https://github.com/thu-ml/prolificdreamer)
- **[NeurIPS 2023]** Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models. [[Paper]](https://arxiv.org/pdf/2305.18455) [[Code]](https://github.com/pkulwj1994/diff_instruct)
- **[ICLR 2022]** Progressive Distillation for Fast Sampling of Diffusion Models. [[Paper]](https://openreview.net/pdf?id=TIdIXIpzhoI) [[Code]](https://github.com/google-research/google-research/tree/master/diffusion_distillation)
- **[Arxiv 2021.01]** Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed. [[Paper]](https://arxiv.org/pdf/2101.02388) [[Code]](https://github.com/tcl9876/Denoising_Student)


### Compression
#### Quantization
- **[NeurIPS 2024]** BitsFusion: 1.99 bits Weight Quantization of Diffusion Model. [[Paper]](https://arxiv.org/pdf/2406.04333)
- **[ICLR 2024]** EfficientDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models. [[Paper]](https://arxiv.org/pdf/2310.03270) [[Code]](https://github.com/ThisisBillhe/EfficientDM)
- **[CVPR 2023]** Post-training Quantization on Diffusion Models. [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shang_Post-Training_Quantization_on_Diffusion_Models_CVPR_2023_paper.pdf) [[Code]](https://github.com/42Shawn/PTQ4DM)
- **[ICCV 2023]** Q-Diffusion: Quantizing Diffusion Models. [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Q-Diffusion_Quantizing_Diffusion_Models_ICCV_2023_paper.pdf) [[Code]](https://github.com/Xiuyu-Li/q-diffusion)
- **[NeurIPS 2023]** Leveraging Early-Stage Robustness in Diffusion Models for Efficient and High-Quality Image Synthesis. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/04261fce1705c4f02f062866717d592a-Paper-Conference.pdf) 
- **[NeurIPS 2023]** PTQD: Accurate Post-Training Quantization for Diffusion Models. [[Paper]](https://arxiv.org/pdf/2305.10657) [[Code]](https://github.com/ziplab/PTQD)
- **[NeurIPS 2023]** Temporal Dynamic Quantization for Diffusion Models. [[Paper]](https://arxiv.org/pdf/2306.02316)
- **[ICLR 2021]** BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction. [[Paper]](https://arxiv.org/pdf/2102.05426) [[Code]](https://github.com/yhhhli/BRECQ)
- **[ICLR 2020]** Learned Step Size Quantization. [[Paper]](https://openreview.net/pdf?id=rkgO66VKDS) 


#### Pruning
- **[CVPRW 2024]** LD-Pruner: Efficient Pruning of Latent Diffusion Models using Task-Agnostic Insights. [[Paper]](https://openaccess.thecvf.com/content/CVPR2024W/EDGE/papers/Castells_LD-Pruner_Efficient_Pruning_of_Latent_Diffusion_Models_using_Task-Agnostic_Insights_CVPRW_2024_paper.pdf)
- **[ICML 2024]** LayerMerge: Neural Network Depth Compression through Layer Pruning and Merging. [[Paper]](https://openreview.net/pdf?id=uDoy7AGvEC) [[Code]](https://github.com/snu-mllab/LayerMerge)
- **[Arxiv 2024.04]** LAPTOP-Diff: Layer Pruning and Normalized Distillation for Compressing Diffusion Models. [[Paper]](https://arxiv.org/pdf/2404.11098) 
- **[NeurIPS 2023]** Structural Pruning for Diffusion Models. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/35c1d69d23bb5dd6b9abcd68be005d5c-Paper-Conference.pdf) [[Code]](https://github.com/VainF/Diff-Pruning)


## 🧑‍💻 System-Level Optimization
#### Hardware-Software Co-Design
- **[FPL 2024]** SDA: Low-Bit Stable Diffusion Acceleration on Edge FPGAs. [[Paper]](https://www.sfu.ca/~zhenman/files/C41-FPL2024-SDA.pdf) [[Code]](https://github.com/Michaela1224/SDA_code)
- **[ISCAS 2024]** A 28.6 mJ/iter Stable Diffusion Processor for Text-to-Image Generation with Patch Similarity-based Sparsity Augmentation and Text-based Mixed-Precision. [[Paper]](https://arxiv.org/pdf/2403.04982)
- **[CVPRW 2023]** Speed Is All You Need: On-Device Acceleration of Large Diffusion Models via GPU-Aware Optimizations. [[Paper]](https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Chen_Speed_Is_All_You_Need_On-Device_Acceleration_of_Large_Diffusion_CVPRW_2023_paper.pdf) [[Project]](https://research.google/blog/speed-is-all-you-need-on-device-acceleration-of-large-diffusion-models-via-gpu-aware-optimizations/)

#### Parallel Computing
- **[CVPR 2024]** DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models. [[Paper]](https://arxiv.org/pdf/2402.19481) [[Code]](https://github.com/mit-han-lab/distrifuser)
- **[Arxiv 2024.07]** SwiftDiffusion: Efficient Diffusion Model Serving with Add-on Modules. [[Paper]](https://arxiv.org/pdf/2407.02031) 
- **[Arxiv 2024.05]** PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models. [[Paper]](https://arxiv.org/pdf/2405.14430) [[Code]](https://github.com/PipeFusion/PipeFusion?tab=readme-ov-file)
- **[MLSys 2024]** DiffusionPipe: Training Large Diffusion Models with Efficient Pipelines. [[Paper]](https://proceedings.mlsys.org/paper_files/paper/2024/file/45c1f6a8cbf2da59ebf2c802b4f742cd-Paper-Conference.pdf)

#### Caching Technique
- **[Arxiv 2025.02]** $ackslash$Delta $-DiT: Accelerating Diffusion Transformers without training via Denoising Property Alignment. [[Paper]](https://openreview.net/pdf?id=pDI03iK5Bf)
- **[NSDI 2024]** Approximate Caching for Efficiently Serving Text-to-Image Diffusion Models. [[Paper]](https://arxiv.org/pdf/2312.04429) [[Code]](https://github.com/mit-han-lab/distrifuser)
- **[CVPR 2024]** DeepCache: Accelerating Diffusion Models for Free. [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_DeepCache_Accelerating_Diffusion_Models_for_Free_CVPR_2024_paper.pdf) [[Code]](https://github.com/horseee/DeepCache)
- **[CVPR 2024]** Cache Me if You Can: Accelerating Diffusion Models through Block Caching. [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Wimbauer_Cache_Me_if_You_Can_Accelerating_Diffusion_Models_through_Block_CVPR_2024_paper.pdf) [[Project]](https://fwmb.github.io/blockcaching/#)
- **[Arxiv 2024.07]** FORA: Fast-Forward Caching in Diffusion Transformer Acceleration. [[Paper]](https://arxiv.org/pdf/2407.01425) [[Code]](https://github.com/prathebaselva/FORA)
- **[Arxiv 2024.06]** Learning-to-Cache: Accelerating Diffusion Transformer via Layer Caching. [[Paper]](https://arxiv.org/pdf/2406.01733) [[Code]](https://github.com/horseee/learning-to-cache)
- **[NeurIPS 2024]** MD-DiT: Step-aware Mixture-of-Depths for Efficient Diffusion Transformers. [[Paper]](https://openreview.net/pdf?id=1jWhiakK7N)


## 🔧 Frameworks
<div align="center">

|                                                                  | Training | Inference | Key Features                                                                 |
|:-------------------------------------------------------------------------:|:--------:|:---------:|:-----------------------------------------------------------------------------|
| FlashAttention [[Code](https://github.com/Dao-AILab/flash-attention)]        |    ✅     |     ✅     | High-efficient attention computation for Diffusion Transformers (DiT)       |
| xFormers [[Code](https://github.com/facebookresearch/xformers)]                    |    ✅     |     ✅     | Memory-efficient attention and modular ops tailored for diffusion Transformer speedups |
| DeepSpeed [[Code](https://github.com/deepspeedai/DeepSpeed)]                  |    ✅     |     ✅     | Scalable distributed training and inference optimizations for large diffusion models |
| OneFlow [[Code](https://github.com/Oneflow-Inc/oneflow)]                      |    ✅     |     ✅     | Compiler-optimized pipeline for faster diffusion model training and sampling |
| Stable-Fast [[Code](https://github.com/chengzeyi/stable-fast)]              |    ❌     |     ✅     | Fast inference optimization for Diffusers with CUDA and fusion              |
| Onediff [[Code](https://github.com/siliconflow/onediff)]                      |    ❌     |     ✅     | Diffusion-specific acceleration with DeepCache and quantization             |
| DeepCache [[Code](https://github.com/horseee/DeepCache)]                  |    ❌     |     ✅     | Reuses cached diffusion features to speed up inference iterations           |
| TGATE [[Code](https://github.com/HaozheLiu-ST/T-GATE)]                          |    ❌     |     ✅     | Temporal gating to streamline cross-attention in diffusion inference        |
| xDiT [[Code](https://github.com/xdit-project/xDiT?tab=readme-ov-file#cite-us)]                            |    ❌     |     ✅     | Parallel inference engine for Diffusion Transformers                        |

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
