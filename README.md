# Efficient Diffusion Model: A Survey

ddl: September 2024

# Reference:


#### Efficient LLM survey
https://arxiv.org/pdf/2312.03863

#### Efficient Diffusion Model for Vision
https://arxiv.org/pdf/2210.09292

#### Diffusion Models: A Comprehensive Survey of Methods and Applications
https://arxiv.org/pdf/2209.00796

#### An Overview of Diffusion Models: Applications, Guided Generation, Statistical Rates and Optimization
https://arxiv.org/pdf/2404.07771

# Paper List

### Topic
#### Design:
- Classifer guidance
- Discrete vs continuous
- Score matching
- Pyramidal design
- Latent representation

#### Process:
- Training
- Noise distribution
- Mixing
- Scheduling
- Retrieval

#### Efficient Sampling
- SDE Solvers
- ODE Solvers
- Optimized Discretization
- Truncated Diffusion
- Knowledge Distillation

## Undecided Papers

1. LanguageFlow: Advancing Diffusion Language Generation with Probabilistic Flows, NAACL 24 [[Paper\]](https://arxiv.org/pdf/2403.16995)
   1.  ODE-solver -> Language Rectified Flow 用 这个东西去替代SDE | ODE，以解决他在text diffusion model的问题

   2.  Dataset: E2E/NLG/ART
2. Stable Target Field for Reduced Variance Score Estimation in Diffusion Models, ICLR 23 [[Paper\]](https://arxiv.org/pdf/2302.00670)
   1.  Training Process Using STD to enhance SGMs, accelerating training process
3. Learning Energy-Based Models by Cooperative Diffusion Recovery Likelihood, ICLR 24 [[Paper\]](https://arxiv.org/pdf/2309.05153) Cooperative Training Dataset: CIFAR10/ImageNet/Celeb-A

## Algorithm Level

### 1-Efficient Sampling 

#### 1.1-Variance Learning/Noise Scheduling 

1. Learning to Efficiently Sample from Diffusion Probabilistic Models, arXiv [[Paper\]](https://arxiv.org/pdf/2106.03802) Learning-Based Sampling->Score-Based Sampling(We instead view the selection of the inference time schedules as an optimization problem, and introduce an exact dynamic programming algorithm that finds the optimal discrete time schedules for any pre-trained DDPM)为ddpm找到最佳离散安排 Dataset:CIFAR-10/ImageNet 64x64
2. ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting, NIPS 23 [[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/2ac2eac5098dba08208807b65c5851cc-Paper-Conference.pdf), ImageNet
3. Empowering Diffusion Models on the Embedding Space for Text Generation, NAACL 24 [[Paper\]](https://arxiv.org/pdf/2212.09412), 

Noise Rescaling, anchor loss

Dataset: WMT14/WMT16/IWSLT4/Gigaword/QQP/Wiki-Auto/Quasar-T 

1. A Cheaper and Better Diffusion Language Model with Soft-Masked Noise, EMNLP 23 [[Paper\]](https://arxiv.org/pdf/2304.04746) Design a linguistic-informed forward process which adds corruptions to the text through strategically soft-masking to better noise the textual data. Dataset: E2E

#### 1.2-Sampling Scheduling (And Mixing?)

1. Align Your Steps: Optimizing Sampling Schedules in Diffusion Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2404.14507)
   1.  Sample Scheduling Dataset:FFHQ/CIFAR10/ImageNet/WebVid10M
2. Parallel Sampling of Diffusion Models, NIPS 23 [[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/0d1986a61e30e5fa408c81216a616e20-Paper-Conference.pdf) Efficient Sampling -> Parallel Sampling LSUN/Square/PushT/Franka Kitchen
3. Simple Hierarchical Planning with Diffusion, ICLR 24 [[Paper\]](https://arxiv.org/pdf/2401.02644) hierarchical planning Dataset: Maze2D/AntMaze/Gym-MuJoCo/FrankaKitchen
4. Accelerating Parallel Sampling of Diffusion Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.09970) Learning-Based sampling->Score-Based Sampling(accelerates the sampling of diffusion models by parallelizing the autoregressive process) Dataset:ImageNet
5. A Unified Sampling Framework for Solver Searching of Diffusion Probabilistic Models, ICLR 24 [[Paper\]](https://arxiv.org/pdf/2312.07243) Learning-Free sampling->Numerical Solver Optimization Dataset:CIFAR-10/CelebA/ImageNet-64/LSUN-Bedroom
6. PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models, arXiv [[Paper\]](https://arxiv.org/pdf/2405.14430) Keywords: DiT parallel approaches, inference to run on PCIe-linked GPUs
7. Accelerating Guided Diffusion Sampling with Splitting Numerical Methods, ICLR 23 [[Paper\]](https://arxiv.org/pdf/2301.11558) Using operator splitting methods to schedule sampling process Dataset:LSUN/FFHQ
8. Diffusion Glancing Transformer for Parallel Sequence-to-Sequence Learning, NAACL 24 [[Paper\]](https://arxiv.org/pdf/2212.10240)

Keywords: Residual Glancing Sampling

Dataset: QQP/MS-COCO

1. Deep Equilibrium Approaches to Diffusion Models, NIPS 22 [[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2022/file/f7f47a73d631c0410cbc2748a8015241-Paper-Conference.pdf), CIFAR-10/CelebA/LSUN
2. Effective Real Image Editing with Accelerated Iterative Diffusion Inversion, ICCV 23 [[Paper\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Pan_Effective_Real_Image_Editing_with_Accelerated_Iterative_Diffusion_Inversion_ICCV_2023_paper.pdf) 

Efficient Sampling -> blended guidance technique Dataset: AFHQ/COCO

#### 1.3-Learned Posterior Sampling

1. DecompDiff: Diffusion Models with Decomposed Priors for Structure-Based Drug Design, ICML 23 [[Paper\]](https://openreview.net/attachment?id=9qy9DizMlr&name=pdf) Better Design -> decomposing the drug space with prior knowledge Dataset: CrossDocked2020
2. Diffusion Posterior Sampling for Linear Inverse Problem Solving: A Filtering Perspective, ICLR 24 [[Paper\]](https://openreview.net/pdf?id=tplXNcHZs1) Learning-Free Sampling->Score-Based Sampling(leverages sequential Monte Carlo methods to solve the corresponding filtering problem)（filtering posterior sampling）算法使用顺序蒙特卡洛方法来解决过滤问题，这是一种高效的后验采样技术 Dataset:FFHQ-1kvalidation/ImageNet-1k-validation
3. Generalized Deep 3D Shape Prior via Part-Discretized Diffusion Process, CVPR 23 [[Paper\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Generalized_Deep_3D_Shape_Prior_via_Part-Discretized_Diffusion_Process_CVPR_2023_paper.pdf),

Better Design -> P-VQ-VAE to capture local geometric information (Posterior)

Dataset: ShapeNet

#### 1.4-Partial Sampling

1. Leapfrog Diffusion Model for Stochastic Trajectory Prediction, CVPR 23 [[Paper\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Mao_Leapfrog_Diffusion_Model_for_Stochastic_Trajectory_Prediction_CVPR_2023_paper.pdf) Efficient Sampling -> leapfrog initializer replace a large number of small denoising steps Dataset: NBA/NFL/SDD/ETH-UCY
2. SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds, NIPS 23 [[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/41bcc9d3bddd9c90e1f44b29e26d97ff-Paper-Conference.pdf) Knowledge Distillation Dataset: MS-COCO
3. ReDi: Efficient Learning-Free Diffusion Inference via Trajectory Retrieval, ICML 23 [[Paper\]](https://openreview.net/attachment?id=SP01yVIC2o&name=pdf) Learning-Based Sampling->Mixing and Scheduling Strategies Dataset:MS-COCO
4. Data-free Distillation of Diffusion Models with Bootstrapping, ICML 24 [[Paper\]](https://arxiv.org/pdf/2306.05544) Technique: knowledge distillation Comments: this paper trains a time-conditioned model that predicts the output of a pretrained diffusion model teacher given any time step. Such a model can be efficiently trained based on bootstrapping from two consecutive sampled steps. Based on this, it can achieve distill diffusion models into a single step. Datasets: FFHQ/LSUN-Bedroom
5. A Simple Early Exiting Framework for Accelerated Sampling in Diffusion Models, ICML 24 [[Paper\]](https://openreview.net/pdf/6a4f1c506f95b1706b690331beeff65a947fddc6.pdf) Earling Exiting Dataset:ImageNet/CelebA *
6. David helps Goliath: Inference-Time Collaboration Between Small Specialized and Large General Diffusion LMs, NAACL 24 [[Paper\]](https://arxiv.org/pdf/2305.14771)

Sharded Models Across Time-ranges and Early Stopping in Decoding

Dataset: DOLLY

1. On Distillation of Guided Diffusion Models, CVPR 23 [[Paper\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Meng_On_Distillation_of_Guided_Diffusion_Models_CVPR_2023_paper.pdf) Progressive Distillation Dataset: ImageNet/CIFAR-10
2. Learning Stackable and Skippable LEGO Bricks for Efficient, Reconfigurable, and Variable-Resolution Diffusion Modeling, ICLR 24 [[Paper\]](https://arxiv.org/pdf/2310.06389)

Mixing and Scheduling Sampling Strategies -> designing tbreak to selectively skipping sampling steps

CIFAR-10/ImageNet

1. Relay Diffusion: Unifying diffusion process across resolutions for image synthesis, ICLR 24 [[Paper\]](https://arxiv.org/pdf/2309.03350) Technique: feature map dimensionality reduction Comments: This paper proposes relay diffusion. In this case, the diffusion process can now continue when changing the resolution or model architectures. This method can reduce the cost of training and inference. New DIFFUSION model RDM
2. Semi-Implicit Denoising Diffusion Models (SIDDMs), NIPS 23 [[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/3882ca2c952276247fe9a993193b00e4-Paper-Conference.pdf), CIFAR10/CelebA-HQ/ImageNet
3. Directly Fine-Tuning Diffusion Models on Differentiable Rewards, ICLR 24 [[Paper\]](https://arxiv.org/pdf/2309.17400) 

Efficient Sampling -> Tuncates backpropagation to only the last K steps of sampling Dataset: LAION/HPDv2

1. InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation, ICLR 24 [[Paper\]](https://arxiv.org/pdf/2309.06380) Efficient Sampling -> 和上一篇差不多 Rectified Flow Dataset: MS COCO

### 2-Solver

#### 2.1-SDE/ODE Theory

1. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions, ICLR 23 [[Paper\]](https://arxiv.org/pdf/2209.11215) Provides theoretical convergence guarantees for (SGMs) under minimal assumptions Dataset:CIFAR-10/ImageNet 64x64
2. Improved Techniques for Maximum Likelihood Estimation for Diffusion ODEs, ICML 23 [[Paper\]](https://openreview.net/attachment?id=jVR2fF8x8x&name=pdf) Learning-Free sampling->Numerical Solver Optimization Dataset:CIFAR-10/ImageNet-32
3. Gaussian Mixture Solvers for Diffusion Models, NIPS 23 [[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/51373b6499708b6fcc38f1e8f8f5b376-Paper-Conference.pdf) Keywords: stochastic differential equations (SDEs), Gaussian Mixture Solvers
4. Denoising MCMC for Accelerating Diffusion-Based Generative Models, ICML 23 [[Paper\]](https://openreview.net/attachment?id=GOousx8DUL&name=pdf) Learning-Free sampling->Numerical Solver Optimization Dataset:CIFAR11/CelebA-HQ-256/FFHQ-1024
5. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps, NIPS 22 [[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2022/file/260a14acce2a89dad36adc8eefe7c59e-Paper-Conference.pdf), CIFAR-10/CelebA/ImageNet/LSUN
6. Score-Based Generative Modeling through Stochastic Differential Equations, ICLR 21 [[Paper\]](https://arxiv.org/pdf/2011.13456) SDE Solvers/ODE Solvers Dataset: CIFAR-10/LSUN/CelebA-HQ
7. Unifying Bayesian Flow Networks and Diffusion Models through Stochastic Differential Equations, ICML 24 [[Paper\]](https://arxiv.org/pdf/2404.15766) Using BFNs to connect SDEs in diffusion model Dataset:text8/CIFAR-10
8. Diffusion Normalizing Flow, NIPS 21 [[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2021/file/876f1f9954de0aa402d91bb988d12cd4-Paper.pdf)

Solver and Discretization Optimization -> SDE-based Optimization

CIFAR-10/MNIST

1. On the Trajectory Regularity of ODE-based Diffusion Sampling, ICML 24 [[Paper\]](https://arxiv.org/pdf/2405.11326)

Accelerating Sampling Algorithm

Dataset: LSUN Bedroom/CIFAR-10/ImageNet-64/FFHQ

### 3-Architecture Optimization

#### 3.1-DDPM Optimization (Discretization Optimization)

#### 3.2-SGM Optimization

1. FP-Diffusion: Improving Score-based Diffusion Models by Enforcing the Underlying Score Fokker-Planck Equation, ICML 23 [[Paper\]](https://openreview.net/attachment?id=UULcrko6Hk&name=pdf) Learning-Based Sampling-> Score-Based Sampling(derive a corresponding equation called the score FPE that characterizes the noise-conditional scores of the perturbed data densities) 提出fpe score工具，基于sgm，改进score funciton学习过程 Dataset:MNIST/Fashion MNIST/CIFAR-10/ImageNet32
2. Accelerating Score-Based Generative Models with Preconditioned Diffusion Sampling, ECCV 22 [[Paper\]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830001.pdf) Learning-Free Sampling->Score-Based Sampling(leverages matrix preconditioning to alleviate severe performance degradation)预处理采样方法 Dataset:MNIST/CIFAR-10/LSUN/FFHQ
3. Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models, ICML 23 [[Paper\]](https://openreview.net/attachment?id=K1OvMEYEI4&name=pdf) Learning-Based Sampling->Learned Posterior Sampling Dataset:ImageNet/CIFAR-10/CelebA/FFHQ
4. Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models, ICLR 22 [[Paper\]](https://arxiv.org/pdf/2201.06503) Efficient Sampling -> Score-Based Sampling Dataset: CIFAR10/ImageNet
5. Discrete Predictor-Corrector Diffusion Models for Image Synthesis, ICLR 23 [[Paper\]](https://openreview.net/pdf?id=VM8batVBWvg)

Solver and Discretization Optimization -> STF(containing ODE and SDE) to accelerate sampling steps 

### 4-Latent Diffusion Optimization

1. Fast Timing-Conditioned Latent Audio Diffusion, ICML 24 [[Paper\]](https://www.arxiv.org/pdf/2402.04825), MusicCaps Latent Diffusion Dataset: MusicCaps/AudioCaps
2. AudioLDM: Text-to-Audio Generation with Latent Diffusion Models, ICML 23 [[Paper\]](https://openreview.net/attachment?id=6BhipYkaSV&name=pdf), AudioCaps Latent Diffusion Dataset: AudioSet/AudioCaps/Freesound/BBC Sound Effect library
3. Executing Your Commands via Motion Diffusion in Latent Space, CVPR 23 [[Paper\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Executing_Your_Commands_via_Motion_Diffusion_in_Latent_Space_CVPR_2023_paper.pdf), HumanML3D/KIT/HumanAct12/UESTC Latent Diffusion Dataset: HumanML3D/KIT/AMASS/HumanAct12/UESTC
4. Efficient Video Diffusion Models via Content-Frame Motion-Latent Decomposition, ICLR 24 [[Paper\]](https://arxiv.org/pdf/2403.14148)  Latent Diffusion Dataset: UCF-101/WebVid-10M/MSR-VTT
5. Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space, ICLR 24 [[Paper\]](https://arxiv.org/pdf/2310.09656) Tabular Data Synthesis/Latent Diffusion self-construced dataset Dataset: Adult/Default/Shoppers/Magic/Faults/Beijing/News
6. High-Resolution Image Synthesis With Latent Diffusion Models, CVPR 22 [[Paper\]](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) Better Design -> propose the Stable Diffusion model, which achieves diffusion and sample processes in latent space. Latent diff的爹 Datasets: ImageNet/CelebA-HQ/FFHQ/LSUN-Churches/LSUN-Bedrooms
7. Hyperbolic Geometric Latent Diffusion Model for Graph Generation, ICML 24 [[Paper\]](https://arxiv.org/pdf/2405.03188) Latent Diffusion Dataset: SBM/BA/Community/Ego/Barabasi-Albert/Grid/Cora/Citeseer/Polblogs/MUTAG/IMDB-B/PROTEINS/COLLAB
8. Latent 3D Graph Diffusion, ICLR 24 [[Paper\]](https://openreview.net/pdf?id=cXbnGtO0NZ) Latent Diffusion Dataset: ChEMBL/PubChemQC/QM9/Drugs
9. PnP Inversion: Boosting Diffusion-based Editing with 3 Lines of Code, ICLR 24 [[Paper\]](https://openreview.net/pdf?id=FoMZ4ljhVw) Better Design -> a novel technique achieving optimal performance of both branches with just three lines of code. Datasets: PIE-Bench
10. Cross-view Masked Diffusion Transformers for Person Image Synthesis, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.01516) Keywords: Conditional Aggregation Tasks: Pose-guiged Person Image Synthesis
11. Towards Consistent Video Editing with Text-to-Image Diffusion Models, NIPS 23 [[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/b6c05f8254a00709e16fb0fdaae56cd8-Paper-Conference.pdf) Better Design -> Latent Diffusion Model + Attention module optimization Dataset: DAVIS
12. Video Probabilistic Diffusion Models in Projected Latent Space, CVPR 23 [[Paper\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Video_Probabilistic_Diffusion_Models_in_Projected_Latent_Space_CVPR_2023_paper.pdf)

Latent Diffusion -> Dimensionality computation reduction

Dataset: UCF101/SkyTimelapse

1. Conditional Image-to-Video Generation With Latent Flow Diffusion Models, CVPR 23 [[Paper\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Ni_Conditional_Image-to-Video_Generation_With_Latent_Flow_Diffusion_Models_CVPR_2023_paper.pdf),

Latent Diffusion

Dataset: MUG

1. Diffusion Autoencoders: Toward a Meaningful and Decodable Representation, CVPR 22 [[Paper\]](https://openaccess.thecvf.com/content/CVPR2022/papers/Preechakul_Diffusion_Autoencoders_Toward_a_Meaningful_and_Decodable_Representation_CVPR_2022_paper.pdf), FFHQ/CelebA-HQ
2. Adapt and Diffuse: Sample-adaptive Reconstruction via Latent Diffusion Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2309.06642) new method改善latent diffusion在latent space 对噪音的估计 Datasets: CelebA-HQ/LSUN-Bedrooms
3. Dimensionality-Varying Diffusion Process, CVPR 23 [[Paper\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Dimensionality-Varying_Diffusion_Process_CVPR_2023_paper.pdf) Better Design -> diminish those inconsequential components and thus use a lower-dimensional signal to represent the source, barely losing information.训练阶段引入低维度信号来优化扩散模型的训练过程。 Datasets: CIFAR-10/LSUN-Bedroom/LSUN-Church/LSUN-Cat/FFHQ
4. Vector Quantized Diffusion Model for Text-to-Image Synthesis, CVPR 22 [[Paper\]](https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_Vector_Quantized_Diffusion_Model_for_Text-to-Image_Synthesis_CVPR_2022_paper.pdf) Better Design -> Inputting Vector Quantization Dataset: CUB-200/Oxford-102/MSCOCO

### 5-Compression

#### 5.1-Knowledge Distillation

1. Bespoke Non-Stationary Solvers for Fast Sampling of Diffusion and Flow Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2403.01329) Knowledge Distillation Dataset:MS-COCO/LibriSpeech/ImageNet-64
2. GENIE: Higher-Order Denoising Diffusion Solvers, NIPS 22 [[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2022/file/c281c5a17ad2e55e1ac1ca825071f991-Paper-Conference.pdf)

Knowledge Distillation

Dataset: CIFAR-10/LSUN,/ImageNet/AFHQv2

1. Diffusion Probabilistic Model Made Slim, CVPR 23 [[Paper\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Diffusion_Probabilistic_Model_Made_Slim_CVPR_2023_paper.pdf) training a small-sized latent diffusion model for light-weight image synthesis. Dataset: ImageNet/MS-COCO

#### 5.2-Quantization

1. Post-Training Quantization on Diffusion Models, CVPR 23 [[Paper\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shang_Post-Training_Quantization_on_Diffusion_Models_CVPR_2023_paper.pdf) Quantization Dataset: ImageNet/CIFAR-10
2. Q-Diffusion: Quantizing Diffusion Models, ICCV 23 [[Paper\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Q-Diffusion_Quantizing_Diffusion_Models_ICCV_2023_paper.pdf) Quantization Dataset: CIFAR-10/LSUN Bedrooms/LSUN Church-Outdoor
3. PTQD: Accurate Post-Training Quantization for Diffusion Models, NIPS 23 [[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/2aab8a76c7e761b66eccaca0927787de-Paper-Conference.pdf), ImageNet/LSUN
4. Binary Latent Diffusion, CVPR 23 [[Paper\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Binary_Latent_Diffusion_CVPR_2023_paper.pdf) Quantization Dataset: LSUN Churches/FFHQ/CelebA-HQ/ImageNet-1K
5. DiffFit: Unlocking Transferability of Large Diffusion Models via SimpleParameter-efficient Fine-Tuning, ICCV 23 [[Paper\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Xie_DiffFit_Unlocking_Transferability_of_Large_Diffusion_Models_via_Simple_Parameter-efficient_ICCV_2023_paper.pdf) Datasets: ImageNet
6. Würstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models, ICLR 24 [[Paper\]](https://arxiv.org/pdf/2306.00637) Compression -> Compression Latent Space Dataset: COCO-30K
7. Leveraging Early-Stage Robustness in Diffusion Models for Efficient and High-Quality Image Synthesis, NIPS 23 [[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/04261fce1705c4f02f062866717d592a-Paper-Conference.pdf), LSUN

#### 5.3-Pruning

1. Structural Pruning for Diffusion Models, NIPS 23 [[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/35c1d69d23bb5dd6b9abcd68be005d5c-Paper-Conference.pdf), CIFAR-10/CelebA-HQ/LSUN/ImageNet

### 6-Better Design

#### 6.1-Better Architecture(Green Color means Rui is responsible for this section)

1. DiffS2UT: A Semantic Preserving Diffusion Model for Textless Direct Speech-to-Speech Translation, EMNLP 23 [[Paper\]](https://arxiv.org/pdf/2310.17570) by applying the diffusion forward process in the continuous speech representation space, while employing the diffusion backward process in the discrete speech unit space. Dataset: VoxPopuli-S2S/Europarl-ST
2. Infinite Resolution Diffusion with Subsampled Mollified States, ICLR 24 [[Paper\]](https://arxiv.org/pdf/2303.18242) Technique: fast sampling, efficient architecture Comments: This paper introduces a generative diffusion model defined in an infinite dimensional Hilbert space, which can model infinite resolution data. It proposes an efficient multi-scale function-space architecture that operates directly on raw sparse coordinates, coupled with a mollified diffusion process that smooths out irregularities. Datasets: FFHQ/LSUN Church/CelebA-HQ
3. DiffIR: Efficient Diffusion Model for Image Restoration, ICCV 23 [[Paper\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Xia_DiffIR_Efficient_Diffusion_Model_for_Image_Restoration_ICCV_2023_paper.pdf) Datasets: CelebA-HQ, LSUN Bedrooms, Places-Standard，新的架构diffir
4. Wavelet Diffusion Models Are Fast and Scalable Image Generators, CVPR 23[[Paper\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Phung_Wavelet_Diffusion_Models_Are_Fast_and_Scalable_Image_Generators_CVPR_2023_paper.pdf) Technique: wavelet transform, fast sampling, novel diffusion process Comments: Incorporating wavelet transform in the diffusion process. Incorporating wavelet information into feature space through the generator to strengthen the awareness of high-frequency components. Achieving efficiency and better reconstruction results.一个减少采样时间的框架，不依赖于特定diffusion model  Datasets: CIFAR-10/STL-10/CelebA-HQ/LSUN-Church
5. Fast Ensembling with Diffusion Schrödinger Bridge, ICLR 24 [[Paper\]](https://arxiv.org/pdf/2404.15814)

Solver and Discretization Optimization -> SDE-based Optimization

CIFAR-10/CIFAR-100/TinyImageNet  

1. Non-autoregressive Conditional Diffusion Models for Time Series Prediction, ICML 23 [[Paper\]](https://openreview.net/attachment?id=wZsnZkviro&name=pdf) future mixup and autoregressive initialization. Dataset: NorPool/Caiso/Traffic/Electricity/Weather/Exchange/ETTh1/ETTm1/Wind
2. Fast Sampling of Diffusion Models via Operator Learning, ICML 23 [[Paper\]](https://openreview.net/attachment?id=gWC3Q3pyHe&name=pdf) SDE Solvers/ODE Solvers Dataset: CIFAR-10/ImageNet-64
3. PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis, ICLR 24 [[Paper\]](https://arxiv.org/pdf/2310.00426) Better Design -> Model Training Process Optimization 对像素依赖做优化，以及做了Text-Image的对齐 Dataset: LAION/SAM/JourneyDB
4. Text Diffusion Model with Encoder-Decoder Transformers for Sequence-to-Sequence Generation, NAACL 24 [[Paper\]](https://aclanthology.org/2024.naacl-long.2.pdf)

Better-Design in model architecture encoder-decoder Transformer 

QQP/Wiki-Auto/Quasar-T/CCD/IWSLT14/WMT14

1. IM-3D: Iterative Multiview Diffusion and Reconstruction for High-Quality 3D Generation, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.08682)

Efficient Pipeline?

Dataset: Objaverse 

#### 6.2-Better Algorithm(Red Color means Xiong is responsible for this section)

1. Neural Diffusion Processes, ICML 23 [[Paper\]](https://openreview.net/attachment?id=tV7GSY5GYG&name=pdf) Learning-Based Sampling->Score-Based Sampling 添加了custom attention block提高采样，用于一般地dpm并没有指明ddpm还是sgm Dataset:MNIST/CELEBA
2. Score Regularized Policy Optimization through Diffusion Behavior, ICLR 24 [[Paper\]](https://arxiv.org/pdf/2310.07297) Better Design -> extract an efficient deterministic inference policy from critic models and pretrained diffusion behavior models, leveraging the latter to directly regularize the policy gradient with the behavior distribution’s score function during optimization. Benchmark: BEAR/TD3+BC/IQL
3. Efficient and Degree-Guided Graph Generation via Discrete Diffusion Modeling, ICML 23 [[Paper\]](https://openreview.net/attachment?id=vn9O1N5ZOw&name=pdf) Better Design -> use empty graphs as the convergent distribution/new generative process that only predicts edges between nodes Dataset: Community/Ego/Polblogs/Cora/Road-Minnesota/PPI/QM9
4. Decomposed Diffusion Sampler for Accelerating Large-Scale Inverse Problems, ICLR 24 [[Paper\]](https://arxiv.org/pdf/2303.05754),  Learning-based Sampling Dataset:fastMRI knee/AAPM 256×256
5. Soft Mixture Denoising: Beyond the Expressive Bottleneck of Diffusion Models, ICLR 24 [[Paper\]](https://arxiv.org/pdf/2309.14068) 

## System Level

## Application Level

1. DITTO: Diffusion Inference-Time T-Optimization for Music Generation, ICML 24 [[Paper\]](https://arxiv.org/pdf/2401.12179)

Better Design -> optimizing initial noise latents

Dataset: Wikifonia Lead-Sheet/MusicCaps

Task:Text-to-Music

1. Inf-DiT: Upsampling Any-Resolution Image with Memory-Efficient Diffusion Transformer, arXiv [[Paper\]](https://arxiv.org/pdf/2405.04312)

Task:High-Resolution Image Generation

1. Inserting Anybody in Diffusion Models via Celeb Basis, NIPS 23 [[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/e6d37cc5723e810b793c834bcb6647cf-Paper-Conference.pdf)

Inserting a new module (Celeb Basis)

Dataset: LAION/StyleGAN

Task:Personalized Image Generation

1. Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models, NIPS 22 [[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2022/file/b9603de9e49d0838e53b6c9cf9d06556-Paper-Conference.pdf), LSUN/Cityscapes Keywords: Spatially Sparse Inference, Sparse Incremental Generative Engine, inference acceleration Task:Image Editing

## Benchmark

### Datasets

### Metric Strategy
