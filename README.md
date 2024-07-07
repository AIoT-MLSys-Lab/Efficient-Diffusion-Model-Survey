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

## Tasks

### Label-to-Image

#### Paper List

1. Score-Based Generative Modeling through Stochastic Differential Equations, ICLR 21 [[Paper]](https://arxiv.org/pdf/2011.13456), CIFAR-10
2. Learning Stackable and Skippable LEGO Bricks for Efficient, Reconfigurable, and Variable-Resolution Diffusion Modeling, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.06389), CIFAR-10/ImageNet
3. Fast Ensembling with Diffusion Schrödinger Bridge, ICLR 24 [[Paper]](https://arxiv.org/pdf/2404.15814), CIFAR-10/CIFAR-100/TinyImageNet
4. Diffusion Normalizing Flow, NIPS 21 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/876f1f9954de0aa402d91bb988d12cd4-Paper.pdf), CIFAR-10/MNIST
5. Soft Mixture Denoising: Beyond the Expressive Bottleneck of Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.14068)
6. Stable Target Field for Reduced Variance Score Estimation in Diffusion Models, ICLR 23 [[Paper]](https://arxiv.org/pdf/2302.00670)
7. Discrete Predictor-Corrector Diffusion Models for Image Synthesis, ICLR 23 [[Paper]](https://openreview.net/pdf?id=VM8batVBWvg)
8. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/260a14acce2a89dad36adc8eefe7c59e-Paper-Conference.pdf), CIFAR-10/CelebA/ImageNet/LSUN
9. GENIE: Higher-Order Denoising Diffusion Solvers, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/c281c5a17ad2e55e1ac1ca825071f991-Paper-Conference.pdf), CIFAR-10/LSUN,/ImageNet/AFHQv2
10. Deep Equilibrium Approaches to Diffusion Models, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/f7f47a73d631c0410cbc2748a8015241-Paper-Conference.pdf), CIFAR-10/CelebA/LSUN
11. Leveraging Early-Stage Robustness in Diffusion Models for Efficient and High-Quality Image Synthesis, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/04261fce1705c4f02f062866717d592a-Paper-Conference.pdf), LSUN
12. PTQD: Accurate Post-Training Quantization for Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/2aab8a76c7e761b66eccaca0927787de-Paper-Conference.pdf), ImageNet/LSUN
13. ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/2ac2eac5098dba08208807b65c5851cc-Paper-Conference.pdf), ImageNet
14. Structural Pruning for Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/35c1d69d23bb5dd6b9abcd68be005d5c-Paper-Conference.pdf), CIFAR-10/CelebA-HQ/LSUN/ImageNet
15. Semi-Implicit Denoising Diffusion Models (SIDDMs), NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/3882ca2c952276247fe9a993193b00e4-Paper-Conference.pdf), CIFAR10/CelebA-HQ/ImageNet
16. Post-Training Quantization on Diffusion Models, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shang_Post-Training_Quantization_on_Diffusion_Models_CVPR_2023_paper.pdf)  
Quantization  
Dataset: ImageNet/CIFAR-10  
17. Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models, ICLR 22 [[Paper]](https://arxiv.org/pdf/2201.06503)   Efficient Sampling -> theoretical?  
Dataset: CIFAR10/ImageNet  
18. On Distillation of Guided Diffusion Models, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Meng_On_Distillation_of_Guided_Diffusion_Models_CVPR_2023_paper.pdf)  
Distillation  
Dataset: ImageNet/CIFAR-10  
19. Binary Latent Diffusion, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Binary_Latent_Diffusion_CVPR_2023_paper.pdf)  
Quantization  
Dataset: LSUN Churches/FFHQ/CelebA-HQ/ImageNet-1K  
20. Q-Diffusion: Quantizing Diffusion Models, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Q-Diffusion_Quantizing_Diffusion_Models_ICCV_2023_paper.pdf)  
Quantization   
Dataset: CIFAR-10/LSUN Bedrooms/LSUN
Church-Outdoor  
21. Score Approximation, Estimation and Distribution Recovery of Diffusion Models on Low-Dimensional Data, ICML 23 [[Paper]](https://openreview.net/attachment?id=KB4mLiuoEX&name=pdf)  
Theoretical?  
Dataset: No  
22. Learning Energy-Based Models by Cooperative Diffusion Recovery Likelihood, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.05153)  
Cooperative Training  
Dataset: CIFAR10/ImageNet/Celeb-A  
23. 

### Image-to-Image

#### Paper List
1. DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation, CVPR 22 [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_DiffusionCLIP_Text-Guided_Diffusion_Models_for_Robust_Image_Manipulation_CVPR_2022_paper.pdf), CelebA-HQ/AFHQ-Dog/LSUN-Bedroom/LSUN-Church
2. Diffusion Autoencoders: Toward a Meaningful and Decodable Representation, CVPR 22 [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Preechakul_Diffusion_Autoencoders_Toward_a_Meaningful_and_Decodable_Representation_CVPR_2022_paper.pdf), FFHQ/CelebA-HQ
3. Continuous-Multiple Image Outpainting in One-Step via Positional Query and A Diffusion-based Approach, ICLR 24 [[Paper]](https://arxiv.org/pdf/2401.15652), Scenery
4. Alleviating Exposure Bias in Diffusion Models through Sampling with Shifted Time Steps, ICLR 24 [[Paper]](https://arxiv.org/pdf/2305.15583), CIFAR-10/CelebA 
5. PnP Inversion: Boosting Diffusion-based Editing with 3 Lines of Code, ICLR 24 [[Paper]](https://openreview.net/pdf?id=FoMZ4ljhVw), PIE-Bench
6. Dimensionality-Varying Diffusion Process, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Dimensionality-Varying_Diffusion_Process_CVPR_2023_paper.pdf), CIFAR-10/LSUN-Bedroom/LSUN-Church/LSUN-Cat/FFHQ
7. DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-efficient Fine-Tuning, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Xie_DiffFit_Unlocking_Transferability_of_Large_Diffusion_Models_via_Simple_Parameter-efficient_ICCV_2023_paper.pdf), ImageNet
8. HumanSD: A Native Skeleton-Guided Diffusion Model for Human Image Generation, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Ju_HumanSD_A_Native_Skeleton-Guided_Diffusion_Model_for_Human_Image_Generation_ICCV_2023_paper.pdf), LAION-Human
9. Controllable Person Image Synthesis with Pose-Constrained Latent Diffusion, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Han_Controllable_Person_Image_Synthesis_with_Pose-Constrained_Latent_Diffusion_ICCV_2023_paper.pdf), DeepFashion
10. Adapt and Diffuse: Sample-adaptive Reconstruction via Latent Diffusion Models, ICML 24 [[Paper]](https://arxiv.org/pdf/2309.06642), CelebA-HQ/LSUN-Bedrooms
11. Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency, ICLR 24 [[Paper]](https://arxiv.org/pdf/2307.08123), FFHQ/CelebA-HQ/LSUN-Bedrooms
12. High-Resolution Image Synthesis With Latent Diffusion Models, CVPR 22 [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf), ImageNet/CelebA-HQ/FFHQ/LSUN-Churches/LSUN-Bedrooms
13. Wavelet Diffusion Models Are Fast and Scalable Image Generators, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Phung_Wavelet_Diffusion_Models_Are_Fast_and_Scalable_Image_Generators_CVPR_2023_paper.pdf), CIFAR-10/STL-10/CelebA-HQ/LSUN-Church
14. DiffIR: Efficient Diffusion Model for Image Restoration, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Xia_DiffIR_Efficient_Diffusion_Model_for_Image_Restoration_ICCV_2023_paper.pdf), CelebA-HQ, LSUN Bedrooms, Places-Standard
15. HSR-Diff: Hyperspectral Image Super-Resolution via Conditional Diffusion Models, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_HSR-Diff_Hyperspectral_Image_Super-Resolution_via_Conditional_Diffusion_Models_ICCV_2023_paper.pdf), CAVE/PaviaU/Chikusei/HypSen
16. 3D-aware Image Generation using 2D Diffusion Models, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Xiang_3D-aware_Image_Generation_using_2D_Diffusion_Models_ICCV_2023_paper.pdf), ImageNet/SDIP Dogs/SDIP Elephants/LSUN Horses
17. Data-free Distillation of Diffusion Models with Bootstrapping, ICML 24 [[Paper]](https://arxiv.org/pdf/2306.05544), FFHQ/LSUN-Bedroom
18. NerfDiff: Single-image View Synthesis with NeRF-guided Distillation from 3D-aware Diffusion[[Paper]](https://arxiv.org/pdf/2302.10109), SRN-ShapeNet/Amazon-Berkeley Objects/Clevr3D
19. Relay Diffusion: Unifying diffusion process across resolutions for image synthesis, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.03350)
21. Infinite Resolution Diffusion with Subsampled Mollified States, ICLR 24 [[Paper]](https://arxiv.org/pdf/2303.18242), FFHQ/LSUN Church/CelebA-HQ
22. Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning, ICML 23 [[Paper]](https://openreview.net/attachment?id=LucUrr5kUi&name=pdf)
23. Inf-DiT: Upsampling Any-Resolution Image with Memory-Efficient Diffusion Transformer, arXiv [[Paper]](https://arxiv.org/pdf/2405.04312)
24. PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models, arXiv [[Paper]](https://arxiv.org/pdf/2405.14430)
25. Cross-view Masked Diffusion Transformers for Person Image Synthesis, ICML 24 [[Paper]](https://arxiv.org/pdf/2402.01516)
26. Hierarchical Integration Diffusion Model for Realistic Image Deblurring, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/5cebc89b113920dbff7c79854ba765a3-Paper-Conference.pdf)
27. Gaussian Mixture Solvers for Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/51373b6499708b6fcc38f1e8f8f5b376-Paper-Conference.pdf)
28. Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/b9603de9e49d0838e53b6c9cf9d06556-Paper-Conference.pdf), LSUN/Cityscapes

### Benchmark & Dataset
1. CelebA-HQ
2. AFHQ-Dog
3. LSUN-Bedroom
4. LSUN-Church 
5. LSUN-Cat
6. FFHQ
7. Scenery
8. CIFAR-10
9. PIE-Bench
10. ImageNet
11. STL-10
12. Places-Standard
13. LSUN Horses
14. SDIP Elephants
15. SDIP Dogs
16. Cityscapes
17. LOL
18. VE-LOL-L
19. CAVE
20. PaviaU
21. Chikusei
22. HypSen

### Image-to-Video
#### Paper List
1. Conditional Image-to-Video Generation With Latent Flow Diffusion Models, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Ni_Conditional_Image-to-Video_Generation_With_Latent_Flow_Diffusion_Models_CVPR_2023_paper.pdf), MUG
2. Video Probabilistic Diffusion Models in Projected Latent Space, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Video_Probabilistic_Diffusion_Models_in_Projected_Latent_Space_CVPR_2023_paper.pdf), UCF101/SkyTimelapse

##### Benchmark & Dataset
1. MUG
2. UCF101
3. SkyTimelapse

### Image-to-3D
#### Paper List
1. Make-It-3D: High-fidelity 3D Creation from A Single Image with Diffusion Prior, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Tang_Make-It-3D_High-fidelity_3D_Creation_from_A_Single_Image_with_Diffusion_ICCV_2023_paper.pdf), DTU
2. Viewset Diffusion: (0-)Image-Conditioned 3D Generative Models from 2D Data, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Szymanowicz_Viewset_Diffusion_0-Image-Conditioned_3D_Generative_Models_from_2D_Data_ICCV_2023_paper.pdf), ShapeNet/Minens/CO3D
3. IM-3D: Iterative Multiview Diffusion and Reconstruction for High-Quality 3D Generation, ICML 24 [[Paper]](https://arxiv.org/pdf/2402.08682)

#### Benchmark & Dataset
1. DTU
2. ShapeNet
3. Minens
4. CO3D

### Point Cloud Completion and Generation
#### Paper List
1. Generalized Deep 3D Shape Prior via Part-Discretized Diffusion Process, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Generalized_Deep_3D_Shape_Prior_via_Part-Discretized_Diffusion_Process_CVPR_2023_paper.pdf), ShapeNet

#### Benchmark & Dataset
1.  ShapeNet

### Text-to-Text
#### Paper List
1. Text Diffusion Model with Encoder-Decoder Transformers for Sequence-to-Sequence Generation, NAACL 24 [[Paper]](https://aclanthology.org/2024.naacl-long.2.pdf)QQP/Wiki-Auto/Quasar-T/CCD/IWSLT14/WMT14
2. LanguageFlow: Advancing Diffusion Language Generation with Probabilistic Flows, NAACL 24 [[Paper]](https://arxiv.org/pdf/2403.16995), E2E/NLG/ART
3. Empowering Diffusion Models on the Embedding Space for Text Generation, NAACL 24 [[Paper]](https://arxiv.org/pdf/2212.09412), WMT14/WMT16/IWSLT4/Gigaword/QQP/Wiki-Auto/Quasar-T
4. Diffusion Glancing Transformer for Parallel Sequence-to-Sequence Learning, NAACL 24 [[Paper]](https://arxiv.org/pdf/2212.10240), QQP/MS-COCO
5. David helps Goliath: Inference-Time Collaboration Between Small Specialized and Large General Diffusion LMs, NAACL 24 [[Paper]](https://arxiv.org/pdf/2305.14771)
6. A Cheaper and Better Diffusion Language Model with Soft-Masked Noise, EMNLP 23 [[Paper]](https://arxiv.org/pdf/2304.04746) 

#### Benchmark & Dataset
1. QQP
2. Wiki-Auto
3. Quasar-T
4. CCD
5. IWSLT14
6. WMT14
7. E2E
8. NLG
9. ART
10. WMT16
11. Gigaword
12. MS-COCO
13. C4

### Text-to-Image 
#### Paper List
1. Vector Quantized Diffusion Model for Text-to-Image Synthesis, CVPR 22 [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_Vector_Quantized_Diffusion_Model_for_Text-to-Image_Synthesis_CVPR_2022_paper.pdf)  
Better Design -> Inputting Vector Quantization  
Dataset: CUB-200/Oxford-102/MSCOCO  
2. Diffusion Probabilistic Model Made Slim, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Diffusion_Probabilistic_Model_Made_Slim_CVPR_2023_paper.pdf)  
Knowledge Distillation -> Optimization in Reverse Process  
Dataset: ImageNet/MS-COCO  
3. Effective Real Image Editing with Accelerated Iterative Diffusion Inversion, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Pan_Effective_Real_Image_Editing_with_Accelerated_Iterative_Diffusion_Inversion_ICCV_2023_paper.pdf)
Efficient Sampling -> Optimization in Reverse Process
Dataset: AFHQ/COCO
4. Zero-Shot Contrastive Loss for Text-Guided Diffusion Image Style Transfer, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Zero-Shot_Contrastive_Loss_for_Text-Guided_Diffusion_Image_Style_Transfer_ICCV_2023_paper.pdf)  
Efficient Sampling -> Optimization in Sampling Strategy  
Dataset: FFHQ/CelebA-HQ/ImageNet/LSUN-Church/Wikiart  
5. Würstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2306.00637)
Compression -> Compression  Ration Generation/Three-Stage Architecture  
Dataset: COCO-30K  
6. PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.00426)  
Better Design -> Model Training Process Optimization  
Dataset: LAION/SAM/JourneyDB  
7. Directly Fine-Tuning Diffusion Models on Differentiable Rewards, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.17400)
Efficient Sampling -> Memory Optimization in Gradient Checkpointing  
Dataset: LAION/HPDv2  
8. InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.06380)  
Efficient Sampling -> Inference Sampling OptimizationOne-Step Generation, Rectified Flow  
Dataset: MS COCO  
9. Towards Consistent Video Editing with Text-to-Image Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/b6c05f8254a00709e16fb0fdaae56cd8-Paper-Conference.pdf)  
Better Design -> Fine-coarse Frame Attention Module/Shift-Restricted Temporal Attention Module  
Dataset: DAVIS  
10. Inserting Anybody in Diffusion Models via Celeb Basis, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/e6d37cc5723e810b793c834bcb6647cf-Paper-Conference.pdf)  
Quantization -> Sparse Neural Networks  
Dataset: LAION/StyleGAN  
11. Discrete Contrastive Diffusion for Cross-Modal Music and Image Generation, ICLR 23 [[Paper]](https://arxiv.org/pdf/2206.07771)  
Latent Diffusion  
Dataset: AIST++/TikTok Dance-Music
12. Retrieval-Augmented Diffusion Models, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/62868cc2fc1eb5cdf321d05b4b88510c-Paper-Conference.pdf)  
Better Design  -> External Data Enhanced  
Dataset: MS-COCO/ImageNet
13. Parallel Sampling of Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/0d1986a61e30e5fa408c81216a616e20-Paper-Conference.pdf)  
Efficient Sampling -> Parallel Sampling  
LSUN/Square/PushT/Franka Kitchen  
14. SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/41bcc9d3bddd9c90e1f44b29e26d97ff-Paper-Conference.pdf)  
Efficient Sampling -> Discrete Latent Space
Dataset: MS-COCO  
15. Diffusion Sampling with Momentum for Mitigating Divergence Artifacts, ICLR 24 [[Paper]](https://arxiv.org/pdf/2307.11118)
Efficient Sampling -> ODE Solvers  
Dataset: ImageNet  

#### Benchmark & Dataset
1. CUB-200
2. COCO
3. AFHQ
4. FFHQ
5. CelebA-HQ
6. ImageNet
7. LSUN- Church
8. CycleGAN dataset
9. LAION
10. Imagenette

### Text-to-Audio

#### Paper List
1. Fast Timing-Conditioned Latent Audio Diffusion, ICML 24 [[Paper]](https://www.arxiv.org/pdf/2402.04825), MusicCaps
2. AudioLDM: Text-to-Audio Generation with Latent Diffusion Models, ICML 23 [[Paper]](https://openreview.net/attachment?id=6BhipYkaSV&name=pdf), AudioCaps
3. DiffS2UT: A Semantic Preserving Diffusion Model for Textless Direct Speech-to-Speech Translation, EMNLP 23 [[Paper]](https://arxiv.org/pdf/2310.17570)
4. DITTO: Diffusion Inference-Time T-Optimization for Music Generation, ICML 24 [[Paper]](https://arxiv.org/pdf/2401.12179)
5. 

#### Benchmark & Dataset
1. MusicCaps
2. AudioCaps

### Text-to-3D
#### Paper List
1. Texture Generation on 3D Meshes with Point-UV Diffusion, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_Texture_Generation_on_3D_Meshes_with_Point-UV_Diffusion_ICCV_2023_paper.pdf), ShapeNet
2. DreamTime: An Improved Optimization Strategy for Diffusion-Guided 3D Generation, ICLR 24 [[Paper]](https://arxiv.org/pdf/2306.12422)

#### Benchmark & Dataset
1. ShapeNet

### Text-to-Motion
#### Paper List
1. Executing Your Commands via Motion Diffusion in Latent Space, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Executing_Your_Commands_via_Motion_Diffusion_in_Latent_Space_CVPR_2023_paper.pdf), HumanML3D/KIT/HumanAct12/UESTC
2. DiffCollage: Parallel Generation of Large Content With Diffusion Models, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_DiffCollage_Parallel_Generation_of_Large_Content_With_Diffusion_Models_CVPR_2023_paper.pdf), HumanML3D

#### Benchmark & Dataset
1. HumanML3D
2. KIT
3. HumanAct12
4. UESTC


### Text-to-Video
#### Paper List
1. DiffTAD: Temporal Action Detection with Proposal Denoising Diffusion, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Nag_DiffTAD_Temporal_Action_Detection_with_Proposal_Denoising_Diffusion_ICCV_2023_paper.pdf), ActivityNet-v1.3/THUMOS14
2. Efficient Video Diffusion Models via Content-Frame Motion-Latent Decomposition, ICLR 24 [[Paper]](https://arxiv.org/pdf/2403.14148)
3. Matryoshka Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.15111)  
Better Design -> progressive training schedule from lower to higher resolutions  
Dataset: ImageNet/CC12M/WebVid-10M

#### Benchmark & Dataset
1. ActivityNet-v1.3
2. THUMOS14

### Temporal Data Modeling
#### Paper List
1. Non-autoregressive Conditional Diffusion Models for Time Series Prediction, ICML 23 [[Paper]](https://openreview.net/attachment?id=wZsnZkviro&name=pdf)

#### Benchmark & Dataset
1. 

### Test-time Adaptation
#### Paper List
1. Back to the Source: Diffusion-Driven Adaptation To Test-Time Corruption, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Gao_Back_to_the_Source_Diffusion-Driven_Adaptation_To_Test-Time_Corruption_CVPR_2023_paper.pdf)  
Better Design -> image guidance and classifier self-ensembling to automatically decide how much to adapt  
Dataset: ImageNet-C (IN-C)  

### Data Generation
#### Paper List
1. Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.09656)  
Tabular Data Synthesis/Latent Diffusion self-construced dataset   
Dataset: Adult/Default/Shoppers/Magic/Faults/Beijing/News  

### Reinforcement Learning
#### Paper List
1. Score Regularized Policy Optimization through Diffusion Behavior, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.07297)  
Better Design -> extract an efficient deterministic inference policy from critic models and pretrained diffusion behavior models  
Benchmark: BEAR/TD3+BC/IQL  

### Efficient Sampling

#### Paper List
1. Decomposed Diffusion Sampler for Accelerating Large-Scale Inverse Problems, ICLR 24 [[Paper]](https://arxiv.org/pdf/2303.05754), fastMRI knee dataset
2. (ODE solvers)Diffusion Sampling with Momentum for Mitigating Divergence Artifacts, ICLR 24 [[Paper]](https://arxiv.org/pdf/2307.11118), COCO
3. Score-Based Generative Modeling through Stochastic Differential Equations, ICLR 21 [[Paper]](https://arxiv.org/pdf/2011.13456)
4. Input Perturbation Reduces Exposure Bias in Diffusion Models, ICML 23 [[Paper]](https://openreview.net/attachment?id=0OcEWSMnSh&name=pdf)
5. Fast Sampling of Diffusion Models via Operator Learning, ICML 23 [[Paper]](https://openreview.net/attachment?id=gWC3Q3pyHe&name=pdf)
6. Denoising MCMC for Accelerating Diffusion-Based Generative Models, ICML 23 [[Paper]](https://openreview.net/attachment?id=GOousx8DUL&name=pdf)
7. Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models, ICML 23 [[Paper]](https://openreview.net/attachment?id=K1OvMEYEI4&name=pdf)
8. ReDi: Efficient Learning-Free Diffusion Inference via Trajectory Retrieval, ICML 23 [[Paper]](https://openreview.net/attachment?id=SP01yVIC2o&name=pdf)
9. Neural Diffusion Processes, ICML 23 [[Paper]](https://openreview.net/attachment?id=tV7GSY5GYG&name=pdf)
10. FP-Diffusion: Improving Score-based Diffusion Models by Enforcing the Underlying Score Fokker-Planck Equation, ICML 23 [[Paper]](https://openreview.net/attachment?id=UULcrko6Hk&name=pdf)
11. Improved Techniques for Maximum Likelihood Estimation for Diffusion ODEs, ICML 23 [[Paper]](https://openreview.net/attachment?id=jVR2fF8x8x&name=pdf)
12. Accelerating Parallel Sampling of Diffusion Models, ICML 24 [[Paper]](https://arxiv.org/pdf/2402.09970)
13. Diffusion Posterior Sampling for Linear Inverse Problem Solving: A Filtering Perspective, ICLR 24 [[Paper]](https://openreview.net/pdf?id=tplXNcHZs1)
14. A Unified Sampling Framework for Solver Searching of Diffusion Probabilistic Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2312.07243)
15. Accelerating Score-Based Generative Models with Preconditioned Diffusion Sampling, ECCV 22 [[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830001.pdf)
16. Learning to Efficiently Sample from Diffusion Probabilistic Models, arXiv [[Paper]](https://arxiv.org/pdf/2106.03802)
17. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions, ICLR 23 [[Paper]](https://arxiv.org/pdf/2209.11215)
18. Directly Denoising Diffusion Models, ICML 24 [[Paper]](https://www.arxiv.org/pdf/2405.13540)
19. Accelerating Guided Diffusion Sampling with Splitting Numerical Methods, ICLR 23 [[Paper]](https://arxiv.org/pdf/2301.11558)
20. A Simple Early Exiting Framework for Accelerated Sampling in Diffusion Models, ICML 24 [[Paper]](https://openreview.net/pdf/6a4f1c506f95b1706b690331beeff65a947fddc6.pdf)
21. Unifying Bayesian Flow Networks and Diffusion Models through Stochastic Differential Equations, ICML 24 [[Paper]](https://arxiv.org/pdf/2404.15766)
22. Align Your Steps: Optimizing Sampling Schedules in Diffusion Models, ICML 24 [[Paper]](https://arxiv.org/pdf/2404.14507)
23. Bespoke Non-Stationary Solvers for Fast Sampling of Diffusion and Flow Models, ICML 24 [[Paper]](https://arxiv.org/pdf/2403.01329)
24. On the Trajectory Regularity of ODE-based Diffusion Sampling, ICML 24 [[Paper]](https://arxiv.org/pdf/2405.11326)
25. 

#### Benchmark & Dataset
1. fastMRI knee dataset
1. COCO

### Trajectory Prediction
#### Paper List
1. Leapfrog Diffusion Model for Stochastic Trajectory Prediction, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Mao_Leapfrog_Diffusion_Model_for_Stochastic_Trajectory_Prediction_CVPR_2023_paper.pdf)
Efficient Sampling -> leapfrog initializer replace a large number of small denoising steps  
Dataset: NBA/NFL/SDD/ETH-UCY  

2. Simple Hierarchical Planning with Diffusion, ICLR 24 [[Paper]](https://arxiv.org/pdf/2401.02644)
Better Design -> hierarchical planning  
Dataset: Maze2D/AntMaze/Gym-MuJoCo/FrankaKitchen  

#### Benchmark & Dataset
1. NBA
2. NFL
3. SDD
4. ETH-UCY

### Graph Generation
#### Paper List
1. Hyperbolic Geometric Latent Diffusion Model for Graph Generation, ICML 24 [[Paper]](https://arxiv.org/pdf/2405.03188)
Better Design -> an improved Gaussian noise generation method  
Dataset: SBM/BA/Community/Ego/Barabasi-Albert/Grid/Cora/Citeseer/Polblogs/MUTAG/IMDB-B/PROTEINS/COLLAB

2. Latent 3D Graph Diffusion, ICLR 24 [[Paper]](https://openreview.net/pdf?id=cXbnGtO0NZ)  
Graph Generation? 3D?  
Latent Diffusion  
Dataset: ChEMBL/PubChemQC/QM9/Drugs  

3. Efficient and Degree-Guided Graph Generation via Discrete Diffusion Modeling, ICML 23 [[Paper]](https://openreview.net/attachment?id=vn9O1N5ZOw&name=pdf)
Better Design -> use empty graphs as the convergent distribution/new generative process that only predicts edges between nodes  
Dataset: Community/Ego/Polblogs/Cora/Road-Minnesota/PPI/QM9  

### Interdisciplinary Applications

#### Paper List
1. DecompDiff: Diffusion Models with Decomposed Priors for Structure-Based Drug Design, ICML 23 [[Paper]](https://openreview.net/attachment?id=9qy9DizMlr&name=pdf)
Molecular Generation  
Better Design -> decomposing the drug space with prior knowledge  
Dataset: CrossDocked2020  

2. Re-Dock: Towards Flexible and Realistic Molecular Docking with Diffusion Bridge, ICML 24 [[Paper]](https://arxiv.org/pdf/2402.11459)  
Molecular Docking  
Better Design  
Dataset: PDBBind v2020


### Undecided
1. 
2. 
3. 
4. 
5. 
6. 
7. 
1. 
2. 
3.  
4.  
5.   
6.  
7.  
8.  Loss-Guided Diffusion Models for Plug-and-Play Controllable Generation, ICML 23 [[Paper]](https://openreview.net/attachment?id=JzZ2xAvCs8&name=pdf)
9.  Better Diffusion Models Further Improve Adversarial Training, ICML 23 [[Paper]](https://openreview.net/attachment?id=1EWPr0ks8I&name=pdf) data generation
10. EfficientDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.03270)
11. Efficient Denoising Diffusion via Probabilistic Masking, ICML 24 [[Paper]](https://openreview.net/pdf?id=lhZEodF8Dn)
12. Q-DM: An Efficient Low-bit Quantized Diffusion Model, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/f1ee1cca0721de55bb35cf28ab95e1b4-Paper-Conference.pdf)
13. Bespoke Non-Stationary Solvers for Fast Sampling of Diffusion and Flow Models, ICML 24 [[Paper]](https://arxiv.org/pdf/2403.01329)
14. EfficientDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.03270)
15. A Variational Perspective on Solving Inverse Problems with Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2305.04391)
16. Denoising Diffusion Step-aware Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.03337)
17. Subspace Diffusion Generative Models, ECCV 22 [[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830274.pdf)
18. DiTFastAttn: Attention Compression for Diffusion Transformer Models, arXiv [[Paper]](https://arxiv.org/pdf/2406.08552)
19. ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation, arXiv [[Paper]](https://arxiv.org/pdf/2406.02540)
20. One-Step Diffusion Distillation via Deep Equilibrium Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/82f05a105c928c10706213952bf0c8b7-Paper-Conference.pdf)
21. Bayesian Power Steering: An Effective Approach for Domain Adaptation of Diffusion Models, ICML 24 [[Paper]](https://arxiv.org/pdf/2406.03683)
22. Generating Behaviorally Diverse Policies with Latent Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/180d4373aca26bd86bf45fc50d1a709f-Paper-Conference.pdf)

## Paper Related to Efficient
1.   David helps Goliath: Inference-Time Collaboration Between Small Specialized and Large General Diffusion LMs, NAACL 24 [[Paper]](https://arxiv.org/pdf/2305.14771)
2.   DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models, ACL 23 [[Paper]](https://arxiv.org/pdf/2211.15029)
3.   DiffusionNER: Boundary Diffusion for Named Entity Recognition, ACL 23 [[Paper]](https://arxiv.org/pdf/2305.13298)
4.   DiffusionDB: A Large-scale Prompt Gallery Dataset for Text-to-Image Generative Models, ACL 23 [[Paper]](https://arxiv.org/pdf/2210.14896)
5.   NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation, ACL 23 [[Paper]](https://arxiv.org/pdf/2303.12346)
6.   A Cheaper and Better Diffusion Language Model with Soft-Masked Noise, EMNLP 23 [[Paper]](https://arxiv.org/pdf/2304.04746)
7.   STINMatch: Semi-Supervised Semantic-Topological Iteration Network for Financial Risk Detection via News Label Diffusion, EMNLP 23 [[Paper]](https://aclanthology.org/2023.emnlp-main.578.pdf)
8.   DiffS2UT: A Semantic Preserving Diffusion Model for Textless Direct Speech-to-Speech Translation, EMNLP 23 [[Paper]](https://arxiv.org/pdf/2310.17570)
9.   ViT-TTS: Visual Text-to-Speech with Scalable Diffusion Transformer, EMNLP 23 [[Paper]](https://arxiv.org/pdf/2305.12708)
10.  Score-Based Generative Modeling through Stochastic Differential Equations, ICLR 21 [[Paper]](https://arxiv.org/pdf/2011.13456)
11.  MOFDiff: Coarse-grained Diffusion for Metal-Organic Framework Design, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.10732)
12.  Single Motion Diffusion, ICLR 24 [[Paper]](https://arxiv.org/pdf/2302.05905)
13.  DreamTime: An Improved Optimization Strategy for Diffusion-Guided 3D Generation, ICLR 24 [[Paper]](https://arxiv.org/pdf/2306.12422)
14.  Directly Fine-Tuning Diffusion Models on Differentiable Rewards, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.17400)
15.  Whole-Song Hierarchical Generation of Symbolic Music Using Cascaded Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2405.09901)
16.  On Error Propagation of Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2308.05021)
17.  Seer: Language Instructed Video Prediction with Latent Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2303.14897)
18.  Simple Hierarchical Planning with Diffusion, ICLR 24 [[Paper]](https://arxiv.org/pdf/2401.02644)
19.  Diffusion-TS: Interpretable Diffusion for General Time Series Generation, ICLR 24 [[Paper]](https://arxiv.org/pdf/2403.01742)
20.  DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2307.02421)
21.  Learning Stackable and Skippable LEGO Bricks for Efficient, Reconfigurable, and Variable-Resolution Diffusion Modeling, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.06389)
22.  Efficient Video Diffusion Models via Content-Frame Motion-Latent Decomposition, ICLR 24 [[Paper]](https://arxiv.org/pdf/2403.14148)
23.  Detecting, Explaining, and Mitigating Memorization in Diffusion Models, ICLR 24 [[Paper]](https://openreview.net/pdf?id=84n3UwkH7b)
24.  JointNet: Extending Text-to-Image Diffusion for Dense Distribution Modeling, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.06347)
25.  Infinite Resolution Diffusion with Subsampled Mollified States, ICLR 24 [[Paper]](https://arxiv.org/pdf/2303.18242)
26.  Fast Ensembling with Diffusion Schrödinger Bridge, ICLR 24 [[Paper]](https://arxiv.org/pdf/2404.15814)
27.  WildFusion: Learning 3D-Aware Latent Diffusion Models in View Space, ICLR 24 [[Paper]](https://arxiv.org/pdf/2311.13570)
28.  InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.06380)
29.  Score Regularized Policy Optimization through Diffusion Behavior, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.07297)
30.  FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.15169)
31.  Navigating the Design Space of Equivariant Diffusion-Based Generative Models for De Novo 3D Molecule Generation, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.17296)
32.  Language Control Diffusion: Efficiently Scaling through Space, Time, and Tasks, ICLR 24 [[Paper]](https://arxiv.org/pdf/2210.15629)
33.  Interpretable Diffusion via Information Decomposition, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.07972)
34.  Maximum Likelihood Training of Score-Based Diffusion Models, NIPS 21 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf), highlights: Image Synthesis, Score-Based, Maximum Likelihood Method
35.  Diffusion Models Beat GANs on Image Synthesis, NIPS 21 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf), highlights: Image Synthesis, GAN comparision
36.  D2C: Diffusion-Decoding Models for Few-Shot Conditional Generation, NIPS 21 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/682e0e796084e163c5ca053dd8573b0c-Paper.pdf), highlights: Few-shot focus
37.  Diffusion Normalizing Flow, NIPS 21 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/876f1f9954de0aa402d91bb988d12cd4-Paper.pdf), highlights: Image Synthesis,
38.  Variational Diffusion Models, NIPS 21 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/b578f2a52a0229873fefc2a4b06377fa-Paper.pdf)
39.  Adaptive Diffusion in Graph Neural Networks, NIPS 21 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/c42af2fa7356818e0389593714f59b52-Paper.pdf)
40.  Local Hyper-Flow Diffusion, NIPS 21 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/e924517087669cf201ea91bd737a4ff4-Paper.pdf)
41.  Thompson Sampling Efficiently Learns to Control Diffusion Processes, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/18c54ed6e0cc390d750f64927dbc4e93-Paper-Conference.pdf)
42.  Diffusion-LM Improves Controllable Text Generation, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/1be5bc25d50895ee656b8c2d9eb89d6a-Paper-Conference.pdf)
43.  Conditional Diffusion Process for Inverse Halftoning, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/2492288f6878e6f99124b362604e58f5-Paper-Conference.pdf)
44.  DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/260a14acce2a89dad36adc8eefe7c59e-Paper-Conference.pdf)
45.  Video Diffusion Models, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/39235c56aef13fb05a6adc95eb9d8d66-Paper-Conference.pdf)
46.  Semantic Diffusion Network for Semantic Segmentation, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/396446770f5e8496ca1feb02079d4fb7-Paper-Conference.pdf)
47.  Diffusion Models as Plug-and-Play Priors, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/5e6cec2a9520708381fe520246018e8b-Paper-Conference.pdf)
48.  Retrieval-Augmented Diffusion Models, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/62868cc2fc1eb5cdf321d05b4b88510c-Paper-Conference.pdf)
49.  CARD: Classification and Regression Diffusion Models, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/72dad95a24fae750f8ab1cb3dab5e58d-Paper-Conference.pdf)
50.  Score-Based Diffusion meets Annealed Importance Sampling, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/86b7128efa3950df7c0f6c0342e6dcc1-Paper-Conference.pdf)
51.  Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/91a85f3fb8f570e6be52b333b5ab017a-Paper-Conference.pdf)
52.  MCVD - Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/944618542d80a63bbec16dfbd2bd689a-Paper-Conference.pdf)
53.  Denoising Diffusion Restoration Models, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/95504595b6169131b6ed6cd72eb05616-Paper-Conference.pdf)
54.  Improving Diffusion Models for Inverse Problems using Manifold Constraints, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/a48e5877c7bf86a513950ab23b360498-Paper-Conference.pdf)
55.  Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/b9603de9e49d0838e53b6c9cf9d06556-Paper-Conference.pdf)
56.  GENIE: Higher-Order Denoising Diffusion Solvers, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/c281c5a17ad2e55e1ac1ca825071f991-Paper-Conference.pdf)
57.  Maximum Likelihood Training of Implicit Nonlinear Diffusion Model, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/d04e47d0fdca09e898885c66b67b1e95-Paper-Conference.pdf)
58.  Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/ec795aeadae0b7d230fa35cbaf04c041-Paper-Conference.pdf)
59.  Deep Equilibrium Approaches to Diffusion Models, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/f7f47a73d631c0410cbc2748a8015241-Paper-Conference.pdf)
60.  Leveraging Early-Stage Robustness in Diffusion Models for Efficient and High-Quality Image Synthesis, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/04261fce1705c4f02f062866717d592a-Paper-Conference.pdf)
61.  From Discrete Tokens to High-Fidelity Audio Using Multi-Band Diffusion, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/054f771d614df12fe8def8ecdbe4e8e1-Paper-Conference.pdf)
62.  PolyDiffuse: Polygonal Shape Reconstruction via Guided Set Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/05f0e2fa003602db2d98ca72b79dec51-Paper-Conference.pdf)
63.  Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/088463cd3126aef2002ffc69da42ec59-Paper-Conference.pdf)
64.  DreamSparse: Escaping from Plato’s Cave with 2D Diffusion Model Given Sparse Views, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/0a003511b09274348b8117f5f3b94c93-Paper-Conference.pdf)
65.  Parallel Sampling of Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/0d1986a61e30e5fa408c81216a616e20-Paper-Conference.pdf)
66.  Direct Diffusion Bridge using Data Consistency for Inverse Problems, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/165b0e600b1721bd59526131eb061092-Paper-Conference.pdf)
67.  Generating Behaviorally Diverse Policies with Latent Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/180d4373aca26bd86bf45fc50d1a709f-Paper-Conference.pdf)
68.  Unsupervised Semantic Correspondence Using Stable Diffusion, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/1a074a28c3a6f2056562d00649ae6416-Paper-Conference.pdf)
69.  Puzzlefusion: Unleashing the Power of Diffusion Models for Spatial Puzzle Solving, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/1e70ac91ad26ba5b24cf11b12a1f90fe-Paper-Conference.pdf)
70.  Star-Shaped Denoising Diffusion Probabilistic Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/1fcefa894924bb1688041b7a26fb8aea-Paper-Conference.pdf)
71.  Graph Denoising Diffusion for Inverse Protein Folding, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/20888d00c5df685de2c09790040e0327-Paper-Conference.pdf)
72.  Drift doesn't Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/22f5d8e689d2a011cd8ead552ed59052-Paper-Conference.pdf)
73.  Diffusion with Forward Models: Solving Stochastic Inverse Problems Without Direct Supervision, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/28e4ee96c94e31b2d040b4521d2b299e-Paper-Conference.pdf)
74.  PTQD: Accurate Post-Training Quantization for Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/2aab8a76c7e761b66eccaca0927787de-Paper-Conference.pdf)
75.  ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/2ac2eac5098dba08208807b65c5851cc-Paper-Conference.pdf)
76.  Structural Pruning for Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/35c1d69d23bb5dd6b9abcd68be005d5c-Paper-Conference.pdf)
77.  Semi-Implicit Denoising Diffusion Models (SIDDMs), NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/3882ca2c952276247fe9a993193b00e4-Paper-Conference.pdf)
78.  SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/41bcc9d3bddd9c90e1f44b29e26d97ff-Paper-Conference.pdf)
79.  Data-Centric Learning from Unlabeled Graphs with Diffusion Model, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/4290cccf23be59e42a575d026ccbeeb8-Paper-Conference.pdf)
80.  Dynamic Tensor Decomposition via Neural Diffusion-Reaction Processes, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/4958a8ad01f524de2ec5274678ffa5a4-Paper-Conference.pdf)

