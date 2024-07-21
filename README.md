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

1. LanguageFlow: Advancing Diffusion Language Generation with Probabilistic Flows, NAACL 24 [[Paper]](https://arxiv.org/pdf/2403.16995)

   ODE-solver -> Using Recited Flow to replace ODE
   Dataset: E2E/NLG/ART

2. Stable Target Field for Reduced Variance Score Estimation in Diffusion Models, ICLR 23 [[Paper]](https://arxiv.org/pdf/2302.00670)
   
   Training Process Using STD to enhance SGMs, accelerating training process
   Dataset: CIFAR-10

3. Learning Energy-Based Models by Cooperative Diffusion Recovery Likelihood, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.05153)
   
   Cooperative Training
   Dataset: CIFAR-10/ImageNet/Celeb-A
   
4. DeepCache: Accelerating Diffusion Models for Free, CVPR 24 [[paper]](https://arxiv.org/abs/2312.00858)
5. Autodiffusion: Training-free optimization of time steps and architectures for automated diffusion model acceleration, ICCV 23 [[paper]](https://arxiv.org/abs/2309.10438)
6. Improving Training Efficiency of Diffusion Models via Multi-Stage Framework and Tailored Multi-Decoder Architecture, CVPR 24 [[paper]] (https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Improving_Training_Efficiency_of_Diffusion_Models_via_Multi-Stage_Framework_and_CVPR_2024_paper.pdf)

## Algorithm Level

### 1-Efficient Sampling 

#### 1.1-Variance Learning/Noise Scheduling 

1. Learning to Efficiently Sample from Diffusion Probabilistic Models, arXiv [[Paper]](https://arxiv.org/pdf/2106.03802)

   Dataset: CIFAR-10/ImageNet 64x64

2. ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/2ac2eac5098dba08208807b65c5851cc-Paper-Conference.pdf)

   Dataset: ImageNet

3. Empowering Diffusion Models on the Embedding Space for Text Generation, NAACL 24 [[Paper]](https://arxiv.org/pdf/2212.09412), 

   Dataset: WMT14/WMT16/IWSLT4/Gigaword/QQP/Wiki-Auto/Quasar-T 

4. A Cheaper and Better Diffusion Language Model with Soft-Masked Noise, EMNLP 23 [[Paper]](https://arxiv.org/pdf/2304.04746)

   Dataset: E2E

#### 1.2-Sampling Scheduling (And Mixing?)

1. Align Your Steps: Optimizing Sampling Schedules in Diffusion Models, ICML 24 [[Paper]](https://arxiv.org/pdf/2404.14507)

   Dataset: FFHQ/CIFAR-10/ImageNet/WebVid10M

2. Parallel Sampling of Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/0d1986a61e30e5fa408c81216a616e20-Paper-Conference.pdf)

   Dataset: LSUN/Square/PushT/Franka Kitchen

3. Simple Hierarchical Planning with Diffusion, ICLR 24 [[Paper]](https://arxiv.org/pdf/2401.02644)

   Dataset: Maze2D/AntMaze/Gym-MuJoCo/FrankaKitchen

4. Accelerating Parallel Sampling of Diffusion Models, ICML 24 [[Paper]](https://arxiv.org/pdf/2402.09970) 

   Dataset: ImageNet

5. A Unified Sampling Framework for Solver Searching of Diffusion Probabilistic Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2312.07243)

   Dataset: CIFAR-10/CelebA/ImageNet-64/LSUN-Bedroom

6. PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models, arXiv [[Paper]](https://arxiv.org/pdf/2405.14430)

   Dataset: COCO Captions 2014

7. Accelerating Guided Diffusion Sampling with Splitting Numerical Methods, ICLR 23 [[Paper]](https://arxiv.org/pdf/2301.11558)

   Dataset: LSUN/FFHQ

8. Diffusion Glancing Transformer for Parallel Sequence-to-Sequence Learning, NAACL 24 [[Paper]](https://arxiv.org/pdf/2212.10240)

   Dataset: QQP/MS-COCO

9. Deep Equilibrium Approaches to Diffusion Models, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/f7f47a73d631c0410cbc2748a8015241-Paper-Conference.pdf)

   Dataset: CIFAR-10/CelebA/LSUN

10. Effective Real Image Editing with Accelerated Iterative Diffusion Inversion, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Pan_Effective_Real_Image_Editing_with_Accelerated_Iterative_Diffusion_Inversion_ICCV_2023_paper.pdf)
    
    Dataset: AFHQ/COCO

#### 1.3-Learned Posterior Sampling

1. DecompDiff: Diffusion Models with Decomposed Priors for Structure-Based Drug Design, ICML 23 [[Paper]](https://openreview.net/attachment?id=9qy9DizMlr&name=pdf)

   Dataset: CrossDocked2020

2. Diffusion Posterior Sampling for Linear Inverse Problem Solving: A Filtering Perspective, ICLR 24 [[Paper]](https://openreview.net/pdf?id=tplXNcHZs1) 

   Dataset: FFHQ-1kvalidation/ImageNet-1k-validation

3. Generalized Deep 3D Shape Prior via Part-Discretized Diffusion Process, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Generalized_Deep_3D_Shape_Prior_via_Part-Discretized_Diffusion_Process_CVPR_2023_paper.pdf),

   Dataset: ShapeNet

#### 1.4-Partial Sampling

1. Leapfrog Diffusion Model for Stochastic Trajectory Prediction, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Mao_Leapfrog_Diffusion_Model_for_Stochastic_Trajectory_Prediction_CVPR_2023_paper.pdf)

   Dataset: NBA/NFL/SDD/ETH-UCY

2. SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/41bcc9d3bddd9c90e1f44b29e26d97ff-Paper-Conference.pdf)

   Dataset: MS-COCO

3. ReDi: Efficient Learning-Free Diffusion Inference via Trajectory Retrieval, ICML 23 [[Paper]](https://openreview.net/attachment?id=SP01yVIC2o&name=pdf)

   Dataset: MS-COCO

4. Data-free Distillation of Diffusion Models with Bootstrapping, ICML 24 [[Paper]](https://arxiv.org/pdf/2306.05544)

   Dataset: FFHQ/LSUN-Bedroom

5. A Simple Early Exiting Framework for Accelerated Sampling in Diffusion Models, ICML 24 [[Paper]](https://openreview.net/pdf/6a4f1c506f95b1706b690331beeff65a947fddc6.pdf)

   Dataset: ImageNet/CelebA

6. David helps Goliath: Inference-Time Collaboration Between Small Specialized and Large General Diffusion LMs, NAACL 24 [[Paper]](https://arxiv.org/pdf/2305.14771)

   Dataset: DOLLY

7. On Distillation of Guided Diffusion Models, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Meng_On_Distillation_of_Guided_Diffusion_Models_CVPR_2023_paper.pdf)

   Dataset: ImageNet/CIFAR-10

8. Learning Stackable and Skippable LEGO Bricks for Efficient, Reconfigurable, and Variable-Resolution Diffusion Modeling, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.06389)

   Dataset: CIFAR-10/ImageNet

9. Relay Diffusion: Unifying diffusion process across resolutions for image synthesis, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.03350)

   Dataset: CelebA-HQ/ImageNet

10. Semi-Implicit Denoising Diffusion Models (SIDDMs), NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/3882ca2c952276247fe9a993193b00e4-Paper-Conference.pdf)

    Dataset: CIFAR-10/CelebA-HQ/ImageNet

11. Directly Fine-Tuning Diffusion Models on Differentiable Rewards, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.17400) 

    Dataset: LAION/HPDv2

12. InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.06380)
  
    Dataset: MS COCO

13. Fast Sampling of Diffusion Models via Operator Learning, ICML 23 [[Paper]](https://openreview.net/attachment?id=gWC3Q3pyHe&name=pdf)
   
    Dataset: CIFAR-10/ImageNet-64

### 2-Solver

#### 2.1-SDE/ODE Theory

1. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions, ICLR 23 [[Paper]](https://arxiv.org/pdf/2209.11215)

   Dataset: CIFAR-10/ImageNet 64x64

2. Improved Techniques for Maximum Likelihood Estimation for Diffusion ODEs, ICML 23 [[Paper]](https://openreview.net/attachment?id=jVR2fF8x8x&name=pdf)

   Dataset: CIFAR-10/ImageNet-32

3. Gaussian Mixture Solvers for Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/51373b6499708b6fcc38f1e8f8f5b376-Paper-Conference.pdf)

   Dataset: CIFAR-10/ImageNet

4. Denoising MCMC for Accelerating Diffusion-Based Generative Models, ICML 23 [[Paper]](https://openreview.net/attachment?id=GOousx8DUL&name=pdf)

   Dataset: CIFAR11/CelebA-HQ-256/FFHQ-1024

5. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/260a14acce2a89dad36adc8eefe7c59e-Paper-Conference.pdf)

   Dataset: CIFAR-10/CelebA/ImageNet/LSUN

6. Score-Based Generative Modeling through Stochastic Differential Equations, ICLR 21 [[Paper]](https://arxiv.org/pdf/2011.13456)

   Dataset: CIFAR-10/LSUN/CelebA-HQ

7. Unifying Bayesian Flow Networks and Diffusion Models through Stochastic Differential Equations, ICML 24 [[Paper]](https://arxiv.org/pdf/2404.15766)

   Dataset: text8/CIFAR-10

8. Diffusion Normalizing Flow, NIPS 21 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/876f1f9954de0aa402d91bb988d12cd4-Paper.pdf)

   Dataset: CIFAR-10/MNIST

9. On the Trajectory Regularity of ODE-based Diffusion Sampling, ICML 24 [[Paper]](https://arxiv.org/pdf/2405.11326)

   Dataset: LSUN Bedroom/CIFAR-10/ImageNet-64/FFHQ

### 3-Architecture Optimization

#### 3.1-DDPM Optimization (Discretization Optimization)

#### 3.2-SGM Optimization

1. FP-Diffusion: Improving Score-based Diffusion Models by Enforcing the Underlying Score Fokker-Planck Equation, ICML 23 [[Paper]](https://openreview.net/attachment?id=UULcrko6Hk&name=pdf)

   Dataset:MNIST/Fashion MNIST/CIFAR-10/ImageNet32

2. Accelerating Score-Based Generative Models with Preconditioned Diffusion Sampling, ECCV 22 [[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830001.pdf)

   Dataset:MNIST/CIFAR-10/LSUN/FFHQ

3. Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models, ICML 23 [[Paper]](https://openreview.net/attachment?id=K1OvMEYEI4&name=pdf)

   Dataset:ImageNet/CIFAR-10/CelebA/FFHQ

4. Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models, ICLR 22 [[Paper]](https://arxiv.org/pdf/2201.06503)

   Dataset: CIFAR-10/ImageNet

5. Discrete Predictor-Corrector Diffusion Models for Image Synthesis, ICLR 23 [[Paper]](https://openreview.net/pdf?id=VM8batVBWvg)

   Dataset: ImageNet/Places2

### 4-Latent Diffusion Optimization

1. Fast Timing-Conditioned Latent Audio Diffusion, ICML 24 [[Paper]](https://www.arxiv.org/pdf/2402.04825)

   Dataset: MusicCaps/AudioCaps

2. AudioLDM: Text-to-Audio Generation with Latent Diffusion Models, ICML 23 [[Paper]](https://openreview.net/attachment?id=6BhipYkaSV&name=pdf)

   Dataset: AudioSet/AudioCaps/Freesound/BBC Sound Effect library
   
3. Executing Your Commands via Motion Diffusion in Latent Space, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Executing_Your_Commands_via_Motion_Diffusion_in_Latent_Space_CVPR_2023_paper.pdf)

   Dataset: HumanML3D/KIT/AMASS/HumanAct12/UESTC

4. Efficient Video Diffusion Models via Content-Frame Motion-Latent Decomposition, ICLR 24 [[Paper]](https://arxiv.org/pdf/2403.14148)

   Dataset: UCF-101/WebVid-10M/MSR-VTT

5. Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.09656)

   Dataset: Adult/Default/Shoppers/Magic/Faults/Beijing/News

6. High-Resolution Image Synthesis With Latent Diffusion Models, CVPR 22 [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
   
   Dataset: ImageNet/CelebA-HQ/FFHQ/LSUN-Churches/LSUN-Bedrooms

7. Hyperbolic Geometric Latent Diffusion Model for Graph Generation, ICML 24 [[Paper]](https://arxiv.org/pdf/2405.03188)
   
   Dataset: SBM/BA/Community/Ego/Barabasi-Albert/Grid/Cora/Citeseer/Polblogs/MUTAG/IMDB-B/PROTEINS/COLLAB

8. Latent 3D Graph Diffusion, ICLR 24 [[Paper]](https://openreview.net/pdf?id=cXbnGtO0NZ)
   
   Dataset: ChEMBL/PubChemQC/QM9/Drugs

9. PnP Inversion: Boosting Diffusion-based Editing with 3 Lines of Code, ICLR 24 [[Paper]](https://openreview.net/pdf?id=FoMZ4ljhVw)

   Dataset: PIE-Bench

10. Cross-view Masked Diffusion Transformers for Person Image Synthesis, ICML 24 [[Paper]](https://arxiv.org/pdf/2402.01516)
    
    Dataset: DeepFashion/ImageNet

11. Towards Consistent Video Editing with Text-to-Image Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/b6c05f8254a00709e16fb0fdaae56cd8-Paper-Conference.pdf)

    Dataset: DAVIS

12. Video Probabilistic Diffusion Models in Projected Latent Space, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Video_Probabilistic_Diffusion_Models_in_Projected_Latent_Space_CVPR_2023_paper.pdf)

    Dataset: UCF101/SkyTimelapse

13. Conditional Image-to-Video Generation With Latent Flow Diffusion Models, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Ni_Conditional_Image-to-Video_Generation_With_Latent_Flow_Diffusion_Models_CVPR_2023_paper.pdf),

    Dataset: MUG

14. Diffusion Autoencoders: Toward a Meaningful and Decodable Representation, CVPR 22 [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Preechakul_Diffusion_Autoencoders_Toward_a_Meaningful_and_Decodable_Representation_CVPR_2022_paper.pdf)

    Dataset: FFHQ/CelebA-HQ

15. Adapt and Diffuse: Sample-adaptive Reconstruction via Latent Diffusion Models, ICML 24 [[Paper]](https://arxiv.org/pdf/2309.06642)

    Dataset: CelebA-HQ/LSUN-Bedrooms

16. Dimensionality-Varying Diffusion Process, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Dimensionality-Varying_Diffusion_Process_CVPR_2023_paper.pdf)

    Dataset: CIFAR-10/LSUN-Bedroom/LSUN-Church/LSUN-Cat/FFHQ

17. Vector Quantized Diffusion Model for Text-to-Image Synthesis, CVPR 22 [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_Vector_Quantized_Diffusion_Model_for_Text-to-Image_Synthesis_CVPR_2022_paper.pdf)

    Dataset: CUB-200/Oxford-102/MSCOCO

### 5-Compression

#### 5.1-Knowledge Distillation

1. Bespoke Non-Stationary Solvers for Fast Sampling of Diffusion and Flow Models, ICML 24 [[Paper]](https://arxiv.org/pdf/2403.01329)

   Dataset: MS-COCO/LibriSpeech/ImageNet-64

2. GENIE: Higher-Order Denoising Diffusion Solvers, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/c281c5a17ad2e55e1ac1ca825071f991-Paper-Conference.pdf)

   Dataset: CIFAR-10/LSUN,/ImageNet/AFHQv2

3. Diffusion Probabilistic Model Made Slim, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Diffusion_Probabilistic_Model_Made_Slim_CVPR_2023_paper.pdf)

   Dataset: ImageNet/MS-COCO

#### 5.2-Quantization

1. Post-Training Quantization on Diffusion Models, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shang_Post-Training_Quantization_on_Diffusion_Models_CVPR_2023_paper.pdf)

   Dataset: ImageNet/CIFAR-10

2. Q-Diffusion: Quantizing Diffusion Models, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Q-Diffusion_Quantizing_Diffusion_Models_ICCV_2023_paper.pdf)

   Dataset: CIFAR-10/LSUN Bedrooms/LSUN Church-Outdoor

3. PTQD: Accurate Post-Training Quantization for Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/2aab8a76c7e761b66eccaca0927787de-Paper-Conference.pdf)

   Dataset: ImageNet/LSUN

4. Binary Latent Diffusion, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Binary_Latent_Diffusion_CVPR_2023_paper.pdf)

   Dataset: LSUN Churches/FFHQ/CelebA-HQ/ImageNet-1K

5. DiffFit: Unlocking Transferability of Large Diffusion Models via SimpleParameter-efficient Fine-Tuning, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Xie_DiffFit_Unlocking_Transferability_of_Large_Diffusion_Models_via_Simple_Parameter-efficient_ICCV_2023_paper.pdf)

   Dataset: ImageNet

6. Würstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2306.00637)

   Dataset: COCO-30K

7. Leveraging Early-Stage Robustness in Diffusion Models for Efficient and High-Quality Image Synthesis, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/04261fce1705c4f02f062866717d592a-Paper-Conference.pdf)

   Dataset: LSUN

#### 5.3-Pruning

1. Structural Pruning for Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/35c1d69d23bb5dd6b9abcd68be005d5c-Paper-Conference.pdf)

   Dataset: CIFAR-10/CelebA-HQ/LSUN/ImageNet

### 6-Better Design

#### 6.1-Better Architecture

1. Infinite Resolution Diffusion with Subsampled Mollified States, ICLR 24 [[Paper]](https://arxiv.org/pdf/2303.18242)

   Dataset: FFHQ/LSUN Church/CelebA-HQ

2. Fast Ensembling with Diffusion Schrödinger Bridge, ICLR 24 [[Paper]](https://arxiv.org/pdf/2404.15814)

   Dataset: CIFAR-10/CIFAR-100/TinyImageNet  

3. Text Diffusion Model with Encoder-Decoder Transformers for Sequence-to-Sequence Generation, NAACL 24 [[Paper]](https://aclanthology.org/2024.naacl-long.2.pdf)

   Dataset: QQP/Wiki-Auto/Quasar-T/CCD/IWSLT14/WMT14

#### 6.2-Better Algorithm

1. Neural Diffusion Processes, ICML 23 [[Paper]](https://openreview.net/attachment?id=tV7GSY5GYG&name=pdf)

   Dataset: MNIST/CELEBA

2. Score Regularized Policy Optimization through Diffusion Behavior, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.07297)
   
   Benchmark: BEAR/TD3+BC/IQL

3. Efficient and Degree-Guided Graph Generation via Discrete Diffusion Modeling, ICML 23 [[Paper]](https://openreview.net/attachment?id=vn9O1N5ZOw&name=pdf)

   Dataset: Community/Ego/Polblogs/Cora/Road-Minnesota/PPI/QM9

4. Decomposed Diffusion Sampler for Accelerating Large-Scale Inverse Problems, ICLR 24 [[Paper]](https://arxiv.org/pdf/2303.05754)

   Dataset: fastMRI knee/AAPM 256×256

5. Soft Mixture Denoising: Beyond the Expressive Bottleneck of Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.14068)

   Dataset: CIGAR-10/LSUN-Conference

## System Level
1. Speed Is All You Need: On-Device Acceleration of Large Diffusion Models via GPU-Aware Optimizations, CVPR 23 [[paper]](https://arxiv.org/abs/2304.11267)
2. DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models, CVPR24 [[paper]](https://arxiv.org/abs/2402.19481)
3. A 28.6 mJ/iter Stable Diffusion Processor for Text-toImage Generation with Patch Similarity-based Sparsity Augmentation and Text-based Mixed-Precision [[paper]] (https://arxiv.org/pdf/2403.04982)
4. 
5. Approximate Caching for Efficiently Serving Text-to-Image Diffusion Models [[paper]](https://www.usenix.org/system/files/nsdi24-agarwal-shubham.pdf)
6. SDA: Low-Bit Stable Diffusion Acceleration on Edge FPGAs, FPL 2024 [[paper]](https://www.sfu.ca/~zhenman/files/C41-FPL2024-SDA.pdf)

## Application Level

1. DITTO: Diffusion Inference-Time T-Optimization for Music Generation, ICML 24 [[Paper]](https://arxiv.org/pdf/2401.12179)

   Dataset: Wikifonia Lead-Sheet/MusicCaps

   Application Task: Text-to-Music

2. Inf-DiT: Upsampling Any-Resolution Image with Memory-Efficient Diffusion Transformer, arXiv [[Paper]](https://arxiv.org/pdf/2405.04312)

   Dataset: HPDv2/DIV2K/LAION-5B/Datacomp

   Application Task: High-Resolution Image Generation

3. Inserting Anybody in Diffusion Models via Celeb Basis, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/e6d37cc5723e810b793c834bcb6647cf-Paper-Conference.pdf)

   Dataset: LAION/StyleGAN

   Application Task: Personalized Image Generation

4. Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/b9603de9e49d0838e53b6c9cf9d06556-Paper-Conference.pdf)

   Dataset: LSUN/Cityscapes

   Application Task: Image Editing

5. DiffS2UT: A Semantic Preserving Diffusion Model for Textless Direct Speech-to-Speech Translation, EMNLP 23 [[Paper]](https://arxiv.org/pdf/2310.17570)

   Dataset: VoxPopuli-S2S/Europarl-ST

   Application Task: Audio-to-Audio

6. DiffIR: Efficient Diffusion Model for Image Restoration, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Xia_DiffIR_Efficient_Diffusion_Model_for_Image_Restoration_ICCV_2023_paper.pdf)

   Dataset: CelebA-HQ, LSUN Bedrooms, Places-Standard

   Application Task: Image Restoration

7. Wavelet Diffusion Models Are Fast and Scalable Image Generators, CVPR 23[[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Phung_Wavelet_Diffusion_Models_Are_Fast_and_Scalable_Image_Generators_CVPR_2023_paper.pdf) 

   Dataset: CIFAR-10/STL-10/CelebA-HQ/LSUN-Church

   Application Task: Image Generation

8.  PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.00426)
   
   Dataset: LAION/SAM/JourneyDB

   Appllication Task: Text-to-Image Generation

9. Non-autoregressive Conditional Diffusion Models for Time Series Prediction, ICML 23 [[Paper]](https://openreview.net/attachment?id=wZsnZkviro&name=pdf)

   Dataset: NorPool/Caiso/Traffic/Electricity/Weather/Exchange/ETTh1/ETTm1/Wind

   Application Task: Time Series Prediction

10. IM-3D: Iterative Multiview Diffusion and Reconstruction for High-Quality 3D Generation, ICML 24 [[Paper]](https://arxiv.org/pdf/2402.08682)

    Dataset: Objaverse
    Application Task: 3D-Object Generation

## Benchmark

### Dataset

### Metric Strategy
