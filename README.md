# Efficient Diffusion Model: A Survey

ddl: September 2024

# Todo List
- 1. Categorizing from the 'modality' perspective

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
10. HSR-Diff: Hyperspectral Image Super-Resolution via Conditional Diffusion Models, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_HSR-Diff_Hyperspectral_Image_Super-Resolution_via_Conditional_Diffusion_Models_ICCV_2023_paper.pdf), CAVE/PaviaU/Chikusei/HypSen
11. 3D-aware Image Generation using 2D Diffusion Models, ICCV 23 [[Paper\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Xiang_3D-aware_Image_Generation_using_2D_Diffusion_Models_ICCV_2023_paper.pdf), ImageNet/SDIP Dogs/SDIP Elephants/LSUN Horses

### Benchmark & Dataset

1. CelebA-HQ CelebA 
2. AFHQ-Dog
3. LSUN-Bedroom
4. LSUN-Church 
5. LSUN-Cat
6. LSUN Horses
7. FFHQ
8. Scenery
9. CIFAR-10
10. PIE-Bench
11. SDIP Elephants
12. SDIP Dogs
13. LIDC
14. Cityscapes
15. LOL
16. VE-LOL-L
17. CAVE
18. PaviaU
19. Chikusei
20. HypSen

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

### Text-to-Image 

#### Paper List
1. Vector Quantized Diffusion Model for Text-to-Image Synthesis, CVPR 22 [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_Vector_Quantized_Diffusion_Model_for_Text-to-Image_Synthesis_CVPR_2022_paper.pdf),CUB-200
2. Diffusion Probabilistic Model Made Slim, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Diffusion_Probabilistic_Model_Made_Slim_CVPR_2023_paper.pdf), ImageNet/MS-COCO
3. Effective Real Image Editing with Accelerated Iterative Diffusion Inversion, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Pan_Effective_Real_Image_Editing_with_Accelerated_Iterative_Diffusion_Inversion_ICCV_2023_paper.pdf), COCO/AFHQ
4. Zero-Shot Contrastive Loss for Text-Guided Diffusion Image Style Transfer, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Zero-Shot_Contrastive_Loss_for_Text-Guided_Diffusion_Image_Style_Transfer_ICCV_2023_paper.pdf), FFHQ/CelebA-HQ/ImageNet/LSUN- Church/CycleGAN dataset
5. Erasing Concepts from Diffusion Models, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Gandikota_Erasing_Concepts_from_Diffusion_Models_ICCV_2023_paper.pdf), COCO/LAION/Imagenette

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

### Text-to-Audio Generation

#### Paper List

1. Fast Timing-Conditioned Latent Audio Diffusion, ICML 24 [[Paper]](https://www.arxiv.org/pdf/2402.04825), MusicCaps
2. AudioLDM: Text-to-Audio Generation with Latent Diffusion Models, ICML 23 [[Paper]](https://openreview.net/attachment?id=6BhipYkaSV&name=pdf), AudioCaps

#### Benchmark & Dataset

1. MusicCaps
2. AudioCaps

### Text-to-3D

#### Paper List

1. Texture Generation on 3D Meshes with Point-UV Diffusion, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_Texture_Generation_on_3D_Meshes_with_Point-UV_Diffusion_ICCV_2023_paper.pdf), ShapeNet
2. 

#### Benchmark & Dataset

1. ShapeNet

### Text-to-Motion

#### Paper List

1. 

#### Benchmark & Dataset

1. 

### Text-to-Video

#### Paper List

1. DiffTAD: Temporal Action Detection with Proposal Denoising Diffusion, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Nag_DiffTAD_Temporal_Action_Detection_with_Proposal_Denoising_Diffusion_ICCV_2023_paper.pdf), ActivityNet-v1.3/THUMOS14
2. 

#### Benchmark & Dataset

1. ActivityNet-v1.3
2. THUMOS14

### Temporal Data Modeling

#### Paper List

1. 

#### Benchmark & Dataset

1. 

### Efficient Sampling->Learning-Based Sampling

#### Paper List
1. Decomposed Diffusion Sampler for Accelerating Large-Scale Inverse Problems, ICLR 24 [[Paper]](https://arxiv.org/pdf/2303.05754), fastMRI knee dataset

#### Benchmark & Dataset
1. fastMRI knee dataset

### Efficient Sampling->Learning-free-Sampling->SDE Solvers 

#### Paper List
1. (ODE solvers)Diffusion Sampling with Momentum for Mitigating Divergence Artifacts, ICLR 24 [[Paper]](https://arxiv.org/pdf/2307.11118), COCO

#### Benchmark & Dataset
1. COCO

### Connections with Other Generative Models

#### Paper List
1. Learning Energy-Based Models by Cooperative Diffusion Recovery Likelihood, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.05153), CIFAR-10

#### Benchmark & Dataset
1. CIFAR-10

### Data with Special Structures → Data with Manifold Structures

#### Paper List
1. Manifold Preserving Guided Diffusion, ICLR 24 [[Paper]](https://arxiv.org/pdf/2311.16424), CIFAR-10

#### Benchmark & Dataset
1. CIFAR-10

### Undecided
1. Post-Training Quantization on Diffusion Models, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shang_Post-Training_Quantization_on_Diffusion_Models_CVPR_2023_paper.pdf)
2. Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models, ICLR 22 [[Paper]](https://arxiv.org/pdf/2201.06503)
3. On Distillation of Guided Diffusion Models, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Meng_On_Distillation_of_Guided_Diffusion_Models_CVPR_2023_paper.pdf)
4. Binary Latent Diffusion, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Binary_Latent_Diffusion_CVPR_2023_paper.pdf)
5. Back to the Source: Diffusion-Driven Adaptation To Test-Time Corruption, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Gao_Back_to_the_Source_Diffusion-Driven_Adaptation_To_Test-Time_Corruption_CVPR_2023_paper.pdf)
6. Solving 3D Inverse Problems Using Pre-Trained 2D Diffusion Models, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Chung_Solving_3D_Inverse_Problems_Using_Pre-Trained_2D_Diffusion_Models_CVPR_2023_paper.pdf)
4. Q-Diffusion: Quantizing Diffusion Models, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Q-Diffusion_Quantizing_Diffusion_Models_ICCV_2023_paper.pdf)

### Human Motion Generation
#### Paper List
1. Executing Your Commands via Motion Diffusion in Latent Space, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Executing_Your_Commands_via_Motion_Diffusion_in_Latent_Space_CVPR_2023_paper.pdf), HumanML3D/KIT/HumanAct12/UESTC
2. DiffCollage: Parallel Generation of Large Content With Diffusion Models, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_DiffCollage_Parallel_Generation_of_Large_Content_With_Diffusion_Models_CVPR_2023_paper.pdf), HumanML3D

#### Benchmark & Dataset
1. HumanML3D
2. KIT
3. HumanAct12
4. UESTC

### Trajectory Prediction
#### Paper List
1. Leapfrog Diffusion Model for Stochastic Trajectory Prediction, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Mao_Leapfrog_Diffusion_Model_for_Stochastic_Trajectory_Prediction_CVPR_2023_paper.pdf), NBA/NFL/SDD/ETH-UCY

#### Benchmark & Dataset
1. NBA
2. NFL
3. SDD
4. ETH-UCY

### Text-to-Audio Generation
#### Paper List
1. Fast Timing-Conditioned Latent Audio Diffusion, ICML 24 [[Paper]](https://www.arxiv.org/pdf/2402.04825)
2. AudioLDM: Text-to-Audio Generation with Latent Diffusion Models, ICML 23 [[Paper]](https://openreview.net/attachment?id=6BhipYkaSV&name=pdf)

#### Benchmark & Dataset
1. MusicCaps
2. AudioCaps

### 3D Graph Generation
#### Paper List
1. Latent 3D Graph Diffusion, ICLR 24 [[Paper]](https://openreview.net/pdf?id=cXbnGtO0NZ), ChEMBL/PubChemQC/QM9/Drugs

#### Benchmark & Dataset
1. ChEMBL
2. PubChemQC
3. QM9
4. Drugs

### Graph Generation
#### Paper List
1. Hyperbolic Geometric Latent Diffusion Model for Graph Generation, ICML 24 [[Paper]](https://arxiv.org/pdf/2405.03188)

#### Benchmark & Dataset
1. Cora
2. Citeseer
3. Polblogs
4. MUTAG
5. IMDB-B
6. PROTEINS

### Image Super Resolution, Inpainting, Restoration, Translation, and Editing
#### Paper List
1. Adapt and Diffuse: Sample-adaptive Reconstruction via Latent Diffusion Models, ICML 24 [[Paper]](https://arxiv.org/pdf/2309.06642), CelebA-HQ/LSUN-Bedrooms
2. Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency, ICLR 24 [[Paper]](https://arxiv.org/pdf/2307.08123), FFHQ/CelebA-HQ/LSUN-Bedrooms
3. High-Resolution Image Synthesis With Latent Diffusion Models, CVPR 22 [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf), ImageNet/CelebA-HQ/FFHQ/LSUN-Churches/LSUN-Bedrooms
4. Wavelet Diffusion Models Are Fast and Scalable Image Generators, CVPR 23 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Phung_Wavelet_Diffusion_Models_Are_Fast_and_Scalable_Image_Generators_CVPR_2023_paper.pdf), CIFAR-10/STL-10/CelebA-HQ/LSUN-Church
5. DiffIR: Efficient Diffusion Model for Image Restoration, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Xia_DiffIR_Efficient_Diffusion_Model_for_Image_Restoration_ICCV_2023_paper.pdf), CelebA-HQ, LSUN Bedrooms, Places-Standard

#### Benchmark & Dataset
1. CelebA-HQ
2. LSUN Bedrooms
3. FFHQ
4. ImageNet
5. LSUN Churches
6. CIFAR-10
7. STL-10
8. Places-Standard

## Paper Related to Efficient
1. 
2. 
3. 
4. 
5. 
6. 
7. 
8. 
9.  
10. 
11. 
12. 
13. 
14. 
15. 
16. 
17. 
18. 
19. 
20. 
21. 
22. 
23. 
24. 
25. 
26. 
27. 
28. 
29. 
30. 
31. 
32. 
33. 
34. 
35. 
36. 
37. 
38. 
39. 
40. 
41. 
42. 
43. 
44. 
45. 
46. 
47. 
48. 
49. 
50. 
51. 
52. 
53. 
54. 
55. 
56. 
57. 
58. 
59. 
60. 
61. Stochastic Segmentation with Conditional Categorical Diffusion Models, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Zbinden_Stochastic_Segmentation_with_Conditional_Categorical_Diffusion_Models_ICCV_2023_paper.pdf)
62. Diff-Retinex: Rethinking Low-light Image Enhancement with A Generative Diffusion Model, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Yi_Diff-Retinex_Rethinking_Low-light_Image_Enhancement_with_A_Generative_Diffusion_Model_ICCV_2023_paper.pdf)
63. Texture Generation on 3D Meshes with Point-UV Diffusion, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_Texture_Generation_on_3D_Meshes_with_Point-UV_Diffusion_ICCV_2023_paper.pdf)
64. Effective Real Image Editing with Accelerated Iterative Diffusion Inversion, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Pan_Effective_Real_Image_Editing_with_Accelerated_Iterative_Diffusion_Inversion_ICCV_2023_paper.pdf)
65. SVDiff: Compact Parameter Space for Diffusion Fine-Tuning, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Han_SVDiff_Compact_Parameter_Space_for_Diffusion_Fine-Tuning_ICCV_2023_paper.pdf)
66. HSR-Diff: Hyperspectral Image Super-Resolution via Conditional Diffusion Models, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_HSR-Diff_Hyperspectral_Image_Super-Resolution_via_Conditional_Diffusion_Models_ICCV_2023_paper.pdf)
67. Zero-Shot Contrastive Loss for Text-Guided Diffusion Image Style Transfer, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Zero-Shot_Contrastive_Loss_for_Text-Guided_Diffusion_Image_Style_Transfer_ICCV_2023_paper.pdf)
68. Erasing Concepts from Diffusion Models, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Gandikota_Erasing_Concepts_from_Diffusion_Models_ICCV_2023_paper.pdf)
69. Make-It-3D: High-fidelity 3D Creation from A Single Image with Diffusion Prior, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Tang_Make-It-3D_High-fidelity_3D_Creation_from_A_Single_Image_with_Diffusion_ICCV_2023_paper.pdf)
70. DiffTAD: Temporal Action Detection with Proposal Denoising Diffusion, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Nag_DiffTAD_Temporal_Action_Detection_with_Proposal_Denoising_Diffusion_ICCV_2023_paper.pdf)
71. TexFusion: Synthesizing 3D Textures with Text-Guided Image Diffusion Models, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_TexFusion_Synthesizing_3D_Textures_with_Text-Guided_Image_Diffusion_Models_ICCV_2023_paper.pdf)
72. Viewset Diffusion: (0-)Image-Conditioned 3D Generative Models from 2D Data, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Szymanowicz_Viewset_Diffusion_0-Image-Conditioned_3D_Generative_Models_from_2D_Data_ICCV_2023_paper.pdf)
73. DiffV2S: Diffusion-Based Video-to-Speech Synthesis with Vision-Guided Speaker Embedding, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Choi_DiffV2S_Diffusion-Based_Video-to-Speech_Synthesis_with_Vision-Guided_Speaker_Embedding_ICCV_2023_paper.pdf)
74. Unsupervised Surface Anomaly Detection with Diffusion Probabilistic Model, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Unsupervised_Surface_Anomaly_Detection_with_Diffusion_Probabilistic_Model_ICCV_2023_paper.pdf)
75. 3D-aware Image Generation using 2D Diffusion Models, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Xiang_3D-aware_Image_Generation_using_2D_Diffusion_Models_ICCV_2023_paper.pdf)
76. StyleDiffusion: Controllable Disentangled Style Transfer via Diffusion Models ICCV 23, [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_StyleDiffusion_Controllable_Disentangled_Style_Transfer_via_Diffusion_Models_ICCV_2023_paper.pdf)
77. Not All Steps are Created Equal: Selective Diffusion Distillation for Image Manipulation, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Not_All_Steps_are_Created_Equal_Selective_Diffusion_Distillation_for_ICCV_2023_paper.pdf)
78. Efficient Diffusion Training via Min-SNR Weighting Strategy, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.pdf)
79. Phasic Content Fusing Diffusion Model with Directional Distribution Consistency for Few-Shot Model Adaption, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Hu_Phasic_Content_Fusing_Diffusion_Model_with_Directional_Distribution_Consistency_for_ICCV_2023_paper.pdf)
80. Sketch and Text Guided Diffusion Model for Colored Point Cloud Generation, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Sketch_and_Text_Guided_Diffusion_Model_for_Colored_Point_Cloud_ICCV_2023_paper.pdf)
81. AutoDiffusion: Training-Free Optimization of Time Steps and Architectures for Automated Diffusion Model Acceleration, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_AutoDiffusion_Training-Free_Optimization_of_Time_Steps_and_Architectures_for_Automated_ICCV_2023_paper.pdf)
82. DiffPose: Multi-hypothesis Human Pose Estimation using Diffusion Models, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Holmquist_DiffPose_Multi-hypothesis_Human_Pose_Estimation_using_Diffusion_Models_ICCV_2023_paper.pdf)
83. DDS2M: Self-Supervised Denoising Diffusion Spatio-Spectral Model for Hyperspectral Image Restoration, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Miao_DDS2M_Self-Supervised_Denoising_Diffusion_Spatio-Spectral_Model_for_Hyperspectral_Image_Restoration_ICCV_2023_paper.pdf)
84. End-to-End Diffusion Latent Optimization Improves Classifier Guidance, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wallace_End-to-End_Diffusion_Latent_Optimization_Improves_Classifier_Guidance_ICCV_2023_paper.pdf)
85. Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Tune-A-Video_One-Shot_Tuning_of_Image_Diffusion_Models_for_Text-to-Video_Generation_ICCV_2023_paper.pdf)
86. Score-Based Diffusion Models as Principled Priors for Inverse Imaging, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Feng_Score-Based_Diffusion_Models_as_Principled_Priors_for_Inverse_Imaging_ICCV_2023_paper.pdf)
87. Your Diffusion Model is Secretly a Zero-Shot Classifier, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.pdf)
88. Preserve Your Own Correlation: A Noise Prior for Video Diffusion Models, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Ge_Preserve_Your_Own_Correlation_A_Noise_Prior_for_Video_Diffusion_ICCV_2023_paper.pdf)
89. Ablating Concepts in Text-to-Image Diffusion Models, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Kumari_Ablating_Concepts_in_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf)
90. Masked Diffusion Transformer is a Strong Image Synthesizer, ICCV 23 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Gao_Masked_Diffusion_Transformer_is_a_Strong_Image_Synthesizer_ICCV_2023_paper.pdf)
91. 
92. 
93. 
94. 
95. 
96. 
97. 
98. 
99. 
100. 
101. 
102. 
103. 
104. 
105. 
106. 
107. 
108. 
109. 
110. 
111. 
112. 
113. 
114. 
115. 
116. 
117. 
118. 
119. 
120. 
121. David helps Goliath: Inference-Time Collaboration Between Small Specialized and Large General Diffusion LMs, NAACL 24 [[Paper]](https://arxiv.org/pdf/2305.14771)
122. DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models, ACL 23 [[Paper]](https://arxiv.org/pdf/2211.15029)
123. DiffusionNER: Boundary Diffusion for Named Entity Recognition, ACL 23 [[Paper]](https://arxiv.org/pdf/2305.13298)
124. DiffusionDB: A Large-scale Prompt Gallery Dataset for Text-to-Image Generative Models, ACL 23 [[Paper]](https://arxiv.org/pdf/2210.14896)
125. NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation, ACL 23 [[Paper]](https://arxiv.org/pdf/2303.12346)
126. A Cheaper and Better Diffusion Language Model with Soft-Masked Noise, EMNLP 23 [[Paper]](https://arxiv.org/pdf/2304.04746)
127. STINMatch: Semi-Supervised Semantic-Topological Iteration Network for Financial Risk Detection via News Label Diffusion, EMNLP 23 [[Paper]](https://aclanthology.org/2023.emnlp-main.578.pdf)
128. DiffS2UT: A Semantic Preserving Diffusion Model for Textless Direct Speech-to-Speech Translation, EMNLP 23 [[Paper]](https://arxiv.org/pdf/2310.17570)
129. ViT-TTS: Visual Text-to-Speech with Scalable Diffusion Transformer, EMNLP 23 [[Paper]](https://arxiv.org/pdf/2305.12708)
130. Score-Based Generative Modeling through Stochastic Differential Equations, ICLR 21 [[Paper]](https://arxiv.org/pdf/2011.13456)
131. MOFDiff: Coarse-grained Diffusion for Metal-Organic Framework Design, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.10732)
132. Single Motion Diffusion, ICLR 24 [[Paper]](https://arxiv.org/pdf/2302.05905)
133. DreamTime: An Improved Optimization Strategy for Diffusion-Guided 3D Generation, ICLR 24 [[Paper]](https://arxiv.org/pdf/2306.12422)
134. Directly Fine-Tuning Diffusion Models on Differentiable Rewards, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.17400)
135. Whole-Song Hierarchical Generation of Symbolic Music Using Cascaded Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2405.09901)
136. On Error Propagation of Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2308.05021)
137. Seer: Language Instructed Video Prediction with Latent Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2303.14897)
138. Simple Hierarchical Planning with Diffusion, ICLR 24 [[Paper]](https://arxiv.org/pdf/2401.02644)
139. Diffusion-TS: Interpretable Diffusion for General Time Series Generation, ICLR 24 [[Paper]](https://arxiv.org/pdf/2403.01742)
140. DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2307.02421)
141. Learning Stackable and Skippable LEGO Bricks for Efficient, Reconfigurable, and Variable-Resolution Diffusion Modeling, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.06389)
142. Efficient Video Diffusion Models via Content-Frame Motion-Latent Decomposition, ICLR 24 [[Paper]](https://arxiv.org/pdf/2403.14148)
143. Detecting, Explaining, and Mitigating Memorization in Diffusion Models, ICLR 24 [[Paper]](https://openreview.net/pdf?id=84n3UwkH7b)
144. JointNet: Extending Text-to-Image Diffusion for Dense Distribution Modeling, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.06347)
145. Infinite Resolution Diffusion with Subsampled Mollified States, ICLR 24 [[Paper]](https://arxiv.org/pdf/2303.18242)
146. Fast Ensembling with Diffusion Schrödinger Bridge, ICLR 24 [[Paper]](https://arxiv.org/pdf/2404.15814)
147. WildFusion: Learning 3D-Aware Latent Diffusion Models in View Space, ICLR 24 [[Paper]](https://arxiv.org/pdf/2311.13570)
148. InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.06380)
149. Score Regularized Policy Optimization through Diffusion Behavior, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.07297)
150. FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.15169)
151. Navigating the Design Space of Equivariant Diffusion-Based Generative Models for De Novo 3D Molecule Generation, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.17296)
152. Language Control Diffusion: Efficiently Scaling through Space, Time, and Tasks, ICLR 24 [[Paper]](https://arxiv.org/pdf/2210.15629)
153. Interpretable Diffusion via Information Decomposition, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.07972)
154. Maximum Likelihood Training of Score-Based Diffusion Models, NIPS 21 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf), highlights: Image Synthesis, Score-Based, Maximum Likelihood Method
155. Diffusion Models Beat GANs on Image Synthesis, NIPS 21 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf), highlights: Image Synthesis, GAN comparision
156. D2C: Diffusion-Decoding Models for Few-Shot Conditional Generation, NIPS 21 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/682e0e796084e163c5ca053dd8573b0c-Paper.pdf), highlights: Few-shot focus
157. Diffusion Normalizing Flow, NIPS 21 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/876f1f9954de0aa402d91bb988d12cd4-Paper.pdf), highlights: Image Synthesis,
158. Variational Diffusion Models, NIPS 21 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/b578f2a52a0229873fefc2a4b06377fa-Paper.pdf)
159. Adaptive Diffusion in Graph Neural Networks, NIPS 21 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/c42af2fa7356818e0389593714f59b52-Paper.pdf)
160. Local Hyper-Flow Diffusion, NIPS 21 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/e924517087669cf201ea91bd737a4ff4-Paper.pdf)
161. Thompson Sampling Efficiently Learns to Control Diffusion Processes, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/18c54ed6e0cc390d750f64927dbc4e93-Paper-Conference.pdf)
162. Diffusion-LM Improves Controllable Text Generation, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/1be5bc25d50895ee656b8c2d9eb89d6a-Paper-Conference.pdf)
163. Conditional Diffusion Process for Inverse Halftoning, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/2492288f6878e6f99124b362604e58f5-Paper-Conference.pdf)
164. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/260a14acce2a89dad36adc8eefe7c59e-Paper-Conference.pdf)
165. Video Diffusion Models, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/39235c56aef13fb05a6adc95eb9d8d66-Paper-Conference.pdf)
166. Semantic Diffusion Network for Semantic Segmentation, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/396446770f5e8496ca1feb02079d4fb7-Paper-Conference.pdf)
167. Diffusion Models as Plug-and-Play Priors, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/5e6cec2a9520708381fe520246018e8b-Paper-Conference.pdf)
168. Retrieval-Augmented Diffusion Models, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/62868cc2fc1eb5cdf321d05b4b88510c-Paper-Conference.pdf)
169. CARD: Classification and Regression Diffusion Models, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/72dad95a24fae750f8ab1cb3dab5e58d-Paper-Conference.pdf)
170. Score-Based Diffusion meets Annealed Importance Sampling, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/86b7128efa3950df7c0f6c0342e6dcc1-Paper-Conference.pdf)
171. Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/91a85f3fb8f570e6be52b333b5ab017a-Paper-Conference.pdf)
172. MCVD - Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/944618542d80a63bbec16dfbd2bd689a-Paper-Conference.pdf)
173. Denoising Diffusion Restoration Models, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/95504595b6169131b6ed6cd72eb05616-Paper-Conference.pdf)
174. Improving Diffusion Models for Inverse Problems using Manifold Constraints, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/a48e5877c7bf86a513950ab23b360498-Paper-Conference.pdf)
175. Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/b9603de9e49d0838e53b6c9cf9d06556-Paper-Conference.pdf)
176. GENIE: Higher-Order Denoising Diffusion Solvers, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/c281c5a17ad2e55e1ac1ca825071f991-Paper-Conference.pdf)
177. Maximum Likelihood Training of Implicit Nonlinear Diffusion Model, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/d04e47d0fdca09e898885c66b67b1e95-Paper-Conference.pdf)
178. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/ec795aeadae0b7d230fa35cbaf04c041-Paper-Conference.pdf)
179. Deep Equilibrium Approaches to Diffusion Models, NIPS 22 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/f7f47a73d631c0410cbc2748a8015241-Paper-Conference.pdf)
180. Leveraging Early-Stage Robustness in Diffusion Models for Efficient and High-Quality Image Synthesis, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/04261fce1705c4f02f062866717d592a-Paper-Conference.pdf)
181. From Discrete Tokens to High-Fidelity Audio Using Multi-Band Diffusion, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/054f771d614df12fe8def8ecdbe4e8e1-Paper-Conference.pdf)
182. PolyDiffuse: Polygonal Shape Reconstruction via Guided Set Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/05f0e2fa003602db2d98ca72b79dec51-Paper-Conference.pdf)
183. Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/088463cd3126aef2002ffc69da42ec59-Paper-Conference.pdf)
184. DreamSparse: Escaping from Plato’s Cave with 2D Diffusion Model Given Sparse Views, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/0a003511b09274348b8117f5f3b94c93-Paper-Conference.pdf)
185. Parallel Sampling of Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/0d1986a61e30e5fa408c81216a616e20-Paper-Conference.pdf)
186. Direct Diffusion Bridge using Data Consistency for Inverse Problems, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/165b0e600b1721bd59526131eb061092-Paper-Conference.pdf)
187. Generating Behaviorally Diverse Policies with Latent Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/180d4373aca26bd86bf45fc50d1a709f-Paper-Conference.pdf)
188. Unsupervised Semantic Correspondence Using Stable Diffusion, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/1a074a28c3a6f2056562d00649ae6416-Paper-Conference.pdf)
189. Puzzlefusion: Unleashing the Power of Diffusion Models for Spatial Puzzle Solving, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/1e70ac91ad26ba5b24cf11b12a1f90fe-Paper-Conference.pdf)
190. Star-Shaped Denoising Diffusion Probabilistic Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/1fcefa894924bb1688041b7a26fb8aea-Paper-Conference.pdf)
191. Graph Denoising Diffusion for Inverse Protein Folding, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/20888d00c5df685de2c09790040e0327-Paper-Conference.pdf)
192. Drift doesn't Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/22f5d8e689d2a011cd8ead552ed59052-Paper-Conference.pdf)
193. Diffusion with Forward Models: Solving Stochastic Inverse Problems Without Direct Supervision, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/28e4ee96c94e31b2d040b4521d2b299e-Paper-Conference.pdf)
194. PTQD: Accurate Post-Training Quantization for Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/2aab8a76c7e761b66eccaca0927787de-Paper-Conference.pdf)
195. ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/2ac2eac5098dba08208807b65c5851cc-Paper-Conference.pdf)
196. Structural Pruning for Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/35c1d69d23bb5dd6b9abcd68be005d5c-Paper-Conference.pdf)
197. Semi-Implicit Denoising Diffusion Models (SIDDMs), NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/3882ca2c952276247fe9a993193b00e4-Paper-Conference.pdf)
198. SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/41bcc9d3bddd9c90e1f44b29e26d97ff-Paper-Conference.pdf)
199. Data-Centric Learning from Unlabeled Graphs with Diffusion Model, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/4290cccf23be59e42a575d026ccbeeb8-Paper-Conference.pdf)
200. Dynamic Tensor Decomposition via Neural Diffusion-Reaction Processes, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/4958a8ad01f524de2ec5274678ffa5a4-Paper-Conference.pdf)
201. Gaussian Mixture Solvers for Diffusion Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/51373b6499708b6fcc38f1e8f8f5b376-Paper-Conference.pdf)
202. Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/5a1a10c2c2c9b9af1514687bc24b8f3d-Paper-Conference.pdf)
203. Hierarchical Integration Diffusion Model for Realistic Image Deblurring, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/5cebc89b113920dbff7c79854ba765a3-Paper-Conference.pdf)
204. Optimal Transport-Guided Conditional Score-Based Diffusion Model, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/72c12e48c6135762f56bf188cd2479d2-Paper-Conference.pdf)
205. Continuous-Time Functional Diffusion Processes, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/75cd262a3fd8e76e37bb7941db141a1d-Paper-Conference.pdf)
206. Improving Diffusion-Based Image Synthesis with Context Prediction, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/7664a7e946a84ac5e97649a967717cf2-Paper-Conference.pdf)
207. Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/80fe51a7d8d0c73ff7439c2a2554ed53-Paper-Conference.pdf)
208. One-Step Diffusion Distillation via Deep Equilibrium Models, NIPS 23 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/82f05a105c928c10706213952bf0c8b7-Paper-Conference.pdf)
209. Nearest Neighbour Score Estimators for Diffusion Generative Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.08018)
210. Cluster-Aware Similarity Diffusion for Instance Retrieval, ICML 24 [[Paper\]](https://arxiv.org/pdf/2406.02343)
211. DITTO: Diffusion Inference-Time T-Optimization for Music Generation, ICML 24 [[Paper\]](https://arxiv.org/pdf/2401.12179)
212. A Simple Early Exiting Framework for Accelerated Sampling in Diffusion Models, ICML 24 [[Paper\]](https://openreview.net/pdf/6a4f1c506f95b1706b690331beeff65a947fddc6.pdf)
213. Improving Diffusion Models for Inverse Problems Using Optimal Posterior Covariance, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.02149)
214. Unifying Bayesian Flow Networks and Diffusion Models through Stochastic Differential Equations, ICML 24 [[Paper\]](https://arxiv.org/pdf/2404.15766)
215. Confronting Reward Overoptimization for Diffusion Models: A Perspective of Inductive and Primacy Biases, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.08552)
216. Neural Diffusion Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2310.08337)
217. Bespoke Non-Stationary Solvers for Fast Sampling of Diffusion and Flow Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2403.01329)
218. IM-3D: Iterative Multiview Diffusion and Reconstruction for High-Quality 3D Generation, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.08682)
219. Align Your Steps: Optimizing Sampling Schedules in Diffusion Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2404.14507)
220. A Dense Reward View on Aligning Text-to-Image Diffusion with Preference, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.08265)
221. Minimax Optimality of Score-based Diffusion Models: Beyond the Density Lower Bound Assumptions, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.15602)
222. On the Trajectory Regularity of ODE-based Diffusion Sampling, ICML 24 [[Paper\]](https://arxiv.org/pdf/2405.11326)
223. Time Series Diffusion in the Frequency Domain, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.05933)
224. Directly Denoising Diffusion Models, ICML 24 [[Paper\]](https://www.arxiv.org/pdf/2405.13540)
225. Re-Dock: Towards Flexible and Realistic Molecular Docking with Diffusion Bridge, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.11459)
226. Consistent Diffusion Meets Tweedie: Training Exact Ambient Diffusion Models with Noisy Data, ICML 24 [[Paper\]](https://arxiv.org/pdf/2404.10177)
227. Listening to the noise: Blind Denoising with Gibbs Diffusion, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.19455)
228. Bayesian Power Steering: An Effective Approach for Domain Adaptation of Diffusion Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2406.03683)
229. Feedback Efficient Online Fine-Tuning of Diffusion Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.16359)
230. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions, ICLR 23 [[Paper]](https://arxiv.org/pdf/2209.11215)
231. Stable Target Field for Reduced Variance Score Estimation in Diffusion Models, ICLR 23 [[Paper]](https://arxiv.org/pdf/2302.00670)
232. Diffusion Posterior Sampling for General Noisy Inverse Problems, ICLR 23 [[Paper]](https://arxiv.org/pdf/2209.14687)
233. Markup-to-Image Diffusion Models with Scheduled Sampling, ICLR 23 [[Paper]](https://arxiv.org/pdf/2210.05147)
234. Discrete Contrastive Diffusion for Cross-Modal Music and Image Generation, ICLR 23 [[Paper]](https://arxiv.org/pdf/2206.07771)
235. DiGress: Discrete Denoising diffusion for graph generation, ICLR 23 [[Paper]](https://arxiv.org/pdf/2209.14734)
236. Diffusion Models Already Have A Semantic Latent Space, ICLR 23 [[Paper]](https://arxiv.org/pdf/2210.10960)
237. SketchKnitter: Vectorized Sketch Generation with Diffusion Models, ICLR 23 [[Paper]](https://openreview.net/pdf?id=4eJ43EN2g6l)
238. Discrete Predictor-Corrector Diffusion Models for Image Synthesis, ICLR 23 [[Paper]](https://openreview.net/pdf?id=VM8batVBWvg)
239. Accelerating Guided Diffusion Sampling with Splitting Numerical Methods, ICLR 23 [[Paper]](https://arxiv.org/pdf/2301.11558)
240. Score-based Continuous-time Discrete Diffusion Models, ICLR 23 [[Paper]](https://arxiv.org/pdf/2211.16750)
241. gDDIM: Generalized denoising diffusion implicit models, ICLR 23 [[Paper]](https://arxiv.org/pdf/2206.05564)
242. Fast Sampling of Diffusion Models with Exponential Integrator, ICLR 23 [[Paper]](https://arxiv.org/pdf/2204.13902)
243. Truncated Diffusion Probabilistic Models and Diffusion-based Adversarial Auto-Encoders, ICLR 23 [[Paper]](https://arxiv.org/pdf/2202.09671)
244. Nearest Neighbour Score Estimators for Diffusion Generative Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.08018)
245. Cluster-Aware Similarity Diffusion for Instance Retrieval, ICML 24 [[Paper\]](https://arxiv.org/pdf/2406.02343)
246. DITTO: Diffusion Inference-Time T-Optimization for Music Generation, ICML 24 [[Paper\]](https://arxiv.org/pdf/2401.12179)
247. A Simple Early Exiting Framework for Accelerated Sampling in Diffusion Models, ICML 24 [[Paper\]](https://openreview.net/pdf/6a4f1c506f95b1706b690331beeff65a947fddc6.pdf)
248. Improving Diffusion Models for Inverse Problems Using Optimal Posterior Covariance, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.02149)
249. Unifying Bayesian Flow Networks and Diffusion Models through Stochastic Differential Equations, ICML 24 [[Paper\]](https://arxiv.org/pdf/2404.15766)
250. Confronting Reward Overoptimization for Diffusion Models: A Perspective of Inductive and Primacy Biases, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.08552)
251. Neural Diffusion Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2310.08337)
252. Bespoke Non-Stationary Solvers for Fast Sampling of Diffusion and Flow Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2403.01329)
253. IM-3D: Iterative Multiview Diffusion and Reconstruction for High-Quality 3D Generation, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.08682)
254. Align Your Steps: Optimizing Sampling Schedules in Diffusion Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2404.14507)
255. A Dense Reward View on Aligning Text-to-Image Diffusion with Preference, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.08265)
256. Minimax Optimality of Score-based Diffusion Models: Beyond the Density Lower Bound Assumptions, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.15602)
257. On the Trajectory Regularity of ODE-based Diffusion Sampling, ICML 24 [[Paper\]](https://arxiv.org/pdf/2405.11326)
258. Time Series Diffusion in the Frequency Domain, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.05933)
259. Directly Denoising Diffusion Models, ICML 24 [[Paper\]](https://www.arxiv.org/pdf/2405.13540)
260. Re-Dock: Towards Flexible and Realistic Molecular Docking with Diffusion Bridge, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.11459)
261. Consistent Diffusion Meets Tweedie: Training Exact Ambient Diffusion Models with Noisy Data, ICML 24 [[Paper\]](https://arxiv.org/pdf/2404.10177)
262. Listening to the noise: Blind Denoising with Gibbs Diffusion, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.19455)
263. Bayesian Power Steering: An Effective Approach for Domain Adaptation of Diffusion Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2406.03683)
264. Feedback Efficient Online Fine-Tuning of Diffusion Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.16359)
265. Variational Schrödinger Diffusion Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2405.04795)
266. Stochastic Conditional Diffusion Models for Robust Semantic Image Synthesis, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.16506)
267. Rolling Diffusion Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.09470)
268. Cross-view Masked Diffusion Transformers for Person Image Synthesis, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.01516)
269. Diffusion Models Demand Contrastive Guidance for Adversarial Purification to Advance, ICML 24 [[Paper\]](https://arxiv.org/pdf/2205.07460)
270. Accelerating Parallel Sampling of Diffusion Models, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.09970)
271. Characteristic Guidance: Non-linear Correction for Diffusion Model at Large Guidance Scale, ICML 24 [[Paper\]](https://arxiv.org/pdf/2312.07586)
272. Efficient Denoising Diffusion via Probabilistic Masking, ICML 24 [[Paper\]](https://openreview.net/pdf?id=lhZEodF8Dn)
273. Accelerating Convergence of Score-Based Diffusion Models, Provably, ICML 24 [[Paper\]](https://arxiv.org/pdf/2403.03852)
274. Diffusion Rejection Sampling, ICML 24 [[Paper\]](https://arxiv.org/pdf/2405.17880)
275. Guidance with Spherical Gaussian Constraint for Conditional Diffusion, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.03201)
276. Score Identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation, ICML 24 [[Paper\]](lhttps://arxiv.org/pdf/2404.04057)
277. Understanding Diffusion Models by Feynman's Path Integral, ICML 24 [[Paper\]](https://arxiv.org/pdf/2404.04057)
278. Diffusion Language Models Are Versatile Protein Learners, ICML 24 [[Paper\]](https://arxiv.org/pdf/2402.18567)
279. Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers, ICML 24 [[Paper\]](https://arxiv.org/pdf/2401.11605)
280. Graph Adversarial Diffusion Convolution, ICML 24 [[Paper\]](https://arxiv.org/pdf/2406.02059v1)
281. Floating Anchor Diffusion Model for Multi-motif Scaffolding, ICML 24 [[Paper\]](https://arxiv.org/pdf/2406.03141)
282. Data-free Distillation of Diffusion Models with Bootstrapping, ICML 24 [[Paper\]](https://arxiv.org/pdf/2306.05544)
283. Restoration-Degradation Beyond Linear Diffusions: A Non-Asymptotic Analysis For DDIM-type Samplers, ICML 23 [[Paper\]](https://openreview.net/attachment?id=GOUgXuLahg&name=pdf)
284. (Efficient Diffusion, Generative Modeling, Uncertainty Quantification)User-defined Event Sampling and Uncertainty Quantification in Diffusion Models for Physical Dynamical Systems, ICML 23 [[Paper\]](https://openreview.net/attachment?id=sdhcjMzhHN&name=pdf)
285. ( Efficient Diffusion, 3D View Synthesis, Neural Radiance Fields (NeRF))NerfDiff: Single-image View Synthesis with NeRF-guided Distillation from 3D-aware Diffusion[[Paper\]](https://arxiv.org/pdf/2302.10109)
286. (Efficient Diffusion, Generative Modeling, Compositional Generation)Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC, ICML 23 [[Paper\]](https://openreview.net/attachment?id=lAXwXjSYum&name=pdf)
287. (Efficient Diffusion, Image Inpainting, Bayesian Framework)Towards Coherent Image Inpainting Using Denoising Diffusion Implicit Models, ICML 23 [[Paper\]](https://openreview.net/attachment?id=17YbAlc1tW&name=pdf)
288. (Efficient Diffusion, Reinforcement Learning, Meta-Learning)MetaDiffuser: Diffusion Model as Conditional Planner for Offline Meta-RL, ICML 23 [[Paper\]](https://openreview.net/attachment?id=IKCk6th595&name=pdf)
289. Loss-Guided Diffusion Models for Plug-and-Play Controllable Generation, ICML 23 [[Paper\]](https://openreview.net/attachment?id=JzZ2xAvCs8&name=pdf)
290. Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning, ICML 23 [[Paper\]](https://openreview.net/attachment?id=LucUrr5kUi&name=pdf)
291. SinDDM: A Single Image Denoising Diffusion Model, ICML 23 [[Paper\]](https://openreview.net/attachment?id=IJ70r39DzS&name=pdf)
292. (Drug Design, Generative Modeling, Efficient Diffusion)DecompDiff: Diffusion Models with Decomposed Priors for Structure-Based Drug Design, ICML 23 [[Paper\]](https://openreview.net/attachment?id=9qy9DizMlr&name=pdf)
293. TabDDPM: Modelling Tabular Data with Diffusion Models, ICML 23 [[Paper\]](https://openreview.net/attachment?id=hTzPqLKBJY&name=pdf)
294. GREAD: Graph Neural Reaction-Diffusion Networks, ICML 23 [[Paper\]](https://openreview.net/attachment?id=LMay53U4ke&name=pdf)
295. (Efficient Diffusion, Generative Modeling, Training Optimization)Input Perturbation Reduces Exposure Bias in Diffusion Models, ICML 23 [[Paper\]](https://openreview.net/attachment?id=0OcEWSMnSh&name=pdf)
296. Denoising MCMC for Accelerating Diffusion-Based Generative Models, ICML 23 [[Paper\]](https://openreview.net/attachment?id=GOousx8DUL&name=pdf)
297. (Efficient Diffusion, Generative Modeling, Machine Learning)Fast Sampling of Diffusion Models via Operator Learning, ICML 23 [[Paper\]](https://openreview.net/attachment?id=gWC3Q3pyHe&name=pdf)
298. Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models, ICML 23 [[Paper\]](https://openreview.net/attachment?id=K1OvMEYEI4&name=pdf)
299. (Efficient Diffusion, Online Decision Making, Bandit Meta-Learning, Thompson Sampling)Thompson Sampling with Diffusion Generative Priors, ICML 23 [[Paper\]](https://openreview.net/attachment?id=Eo7e468wi8&name=pdf)
300. Diffusion Models for Black-Box Optimization, ICML 23 [[Paper\]](https://openreview.net/attachment?id=hQCgc6T15R&name=pdf)
301. (??a new framework for protein structure degign)SE(3) diffusion model with application to protein backbone generation, ICML 23 [[Paper\]](https://openreview.net/attachment?id=m8OUBymxwv&name=pdf)
302. (Efficient Diffusion, Image-to-Image Translation, Manifold Learning, Generative Models)SDDM: Score-Decomposed Diffusion Models on Manifolds for Unpaired Image-to-Image Translation, ICML 23 [[Paper\]](https://openreview.net/attachment?id=J4w91xRPBY&name=pdf)
303. (Efficient Diffusion, Generative Modeling, Machine Learning Optimization)ReDi: Efficient Learning-Free Diffusion Inference via Trajectory Retrieval, ICML 23 [[Paper\]](https://openreview.net/attachment?id=SP01yVIC2o&name=pdf)
304. OMS-DPM: Optimizing the Model Schedule for Diffusion Probabilistic Models, ICML 23 [[Paper\]](https://openreview.net/attachment?id=miv2ZYRd1t&name=pdf)
305. Better Diffusion Models Further Improve Adversarial Training, ICML 23 [[Paper\]](https://openreview.net/attachment?id=1EWPr0ks8I&name=pdf)
306. A Critical Review of Adversarial Robustness in 3D Point Cloud Recognition with Diffusion-Driven Purification, ICML 23 [[Paper\]](https://openreview.net/attachment?id=dwn6o2pYJp&name=pdf)
307. Non-autoregressive Conditional Diffusion Models for Time Series Prediction, ICML 23 [[Paper\]](https://openreview.net/attachment?id=wZsnZkviro&name=pdf)
308. Modeling Temporal Data as Continuous Functions with Stochastic Process Diffusion, ICML 23 [[Paper\]](https://openreview.net/attachment?id=OUWckW2g3j&name=pdf)
309. Neural Diffusion Processes, ICML 23 [[Paper\]](https://openreview.net/attachment?id=tV7GSY5GYG&name=pdf)
310. (Generative Modeling, Stochastic Processes, Machine Learning)A Flexible Diffusion Model, ICML 23 [[Paper\]](
311. AdaptDiffuser: Diffusion Models as Adaptive Self-evolving Planners, ICML 23 [[Paper\]](https://openreview.net/attachment?id=3ETNXs54HB&name=pdf)
312. (Generative Modeling, Efficient Diffusion, Discrete-state Processes)Blackout Diffusion: Generative Diffusion Models in Discrete-State Spaces, ICML 23 [[Paper\]](https://openreview.net/attachment?id=lpc5vlfxp8&name=pdf)
313. Text Generation with Diffusion Language Models: A Pre-training Approach with Continuous Paragraph Denoise, ICML 23 [[Paper\]](https://openreview.net/attachment?id=oY8JcQzahW&name=pdf)
314. MoDiff: Addressing the Atom-Bond Inconsistency Problem in 3D Molecule Diffusion Generation, ICML 23 [[Paper\]](https://openreview.net/attachment?id=gfGLMZR27W&name=pdf)
315. ( Efficient Diffusion, Generative Modeling, Score-based Methods)FP-Diffusion: Improving Score-based Diffusion Models by Enforcing the Underlying Score Fokker-Planck Equation, ICML 23 [[Paper\]](https://openreview.net/attachment?id=UULcrko6Hk&name=pdf)
316. PixelAsParam: A Gradient View on Diffusion Sampling with Guidance, ICML 23 [[Paper\]](https://openreview.net/attachment?id=2q1Whv1kXL&name=pdf)
317. (Efficient Diffusion, Drug Discovery, Generative Modeling)Coarse-to-Fine: A Hierarchical Diffusion Model for Molecule Generation in 3D, ICML 23 [[Paper\]](https://openreview.net/attachment?id=7haEvhb25X&name=pdf)
318. Reflected Diffusion Models, ICML 23 [[Paper\]](https://openreview.net/attachment?id=QcKoLuEO3r&name=pdf)
319. Improved Techniques for Maximum Likelihood Estimation for Diffusion ODEs, ICML 23 [[Paper\]](https://openreview.net/attachment?id=jVR2fF8x8x&name=pdf)
320. GibbsDDRM: A Partially Collapsed Gibbs Sampler for Solving Blind Inverse Problems with Denoising Diffusion Restoration, ICML 23 [[Paper\]](https://openreview.net/attachment?id=4weSHLFgtZ&name=pdf)
321. Improving Adversarial Robustness Through the Contrastive-Guided Diffusion Process, ICML 23 [[Paper\]](https://openreview.net/attachment?id=9iNScYEBWZ&name=pdf)
322. Efficient and Degree-Guided Graph Generation via Discrete Diffusion Modeling, ICML 23 [[Paper\]](https://openreview.net/attachment?id=vn9O1N5ZOw&name=pdf)
323. Diffusion Based Representation Learning, ICML 23 [[Paper\]](https://openreview.net/attachment?id=LCAjuPNJP0&name=pdf)
324. SinFusion: Training Diffusion Models on a Single Image or Video, ICML 23 [[Paper\]](https://openreview.net/attachment?id=9n9NJ4qMV6&name=pdf)
325. (??an algorithm that augments diffusion models with low-dimensional latent variables that capture high-level factors of variation in the data. I)InfoDiffusion: Representation Learning Using Information Maximizing Diffusion Models, ICML 23 [[Paper\]](https://openreview.net/attachment?id=ycZSQdo2F9&name=pdf)
326. Score Approximation, Estimation and Distribution Recovery of Diffusion Models on Low-Dimensional Data, ICML 23 [[Paper\]](https://openreview.net/attachment?id=KB4mLiuoEX&name=pdf)
327. Make-An-Audio: Text-To-Audio Generation with Prompt-Enhanced Diffusion Models, ICML 23 [[Paper\]](https://openreview.net/attachment?id=Srbl4AuTyd&name=pdf)
328. (Efficient Diffusion, High-Resolution Image Synthesis, Generative Modeling)simple diffusion: End-to-end diffusion for high resolution images, ICML 23 [[Paper\]](https://openreview.net/attachment?id=6l9YG3wHA9&name=pdf)
329. MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation, ICML 23 [[Paper\]](https://openreview.net/attachment?id=D4ajVWmgLB&name=pdf)
330. Bidirectional Temporal Diffusion Model for Temporally Consistent Human Animation, ICLR 24 [[Paper]](https://arxiv.org/pdf/2307.00574)
331. Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.09656)
332. Masked Completion via Structured Diffusion with White-Box Transformers, ICLR 24 [[Paper]](https://arxiv.org/pdf/2404.02446)
333. Universal Guidance for Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2302.07121)
334. Elucidating the design space of classifier-guided diffusion generation, ICLR 24 [[Paper]](https://openreview.net/pdf?id=9DXXMXnIGm)
335. Protein-Ligand Interaction Prior for Binding-aware 3D Molecule Diffusion Models, ICLR 24 [[Paper]](https://openreview.net/pdf?id=qH9nrMNTIW)
336. Large-Vocabulary 3D Diffusion Model with Transformer, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.07920)
337. PanoDiffusion: 360-degree Panorama Outpainting via Diffusion, ICLR 24 [[Paper]](https://arxiv.org/pdf/2307.03177)
338. ADDP: Learning General Representations for Image Recognition and Generation with Alternating Denoising Diffusion Process, ICLR 24 [[Paper]](https://arxiv.org/pdf/2306.05423)
339. GeoDiffusion: Text-Prompted Geometric Control for Object Detection Data Generation, ICLR 24 [[Paper]](https://arxiv.org/pdf/2306.04607)
340. Matryoshka Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.15111)
341. Zero-Shot Robotic Manipulation with Pre-Trained Image-Editing Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.10639)
342. Würstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2306.00637)
343. DiffAR: Denoising Diffusion Autoregressive Model for Raw Speech Waveform Generation, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.01381)
344. Don't Play Favorites: Minority Guidance for Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2301.12334)
345. AlignDiff: Aligning Diverse Human Preferences via Behavior-Customisable Diffusion Model, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.02054)
346. Relay Diffusion: Unifying diffusion process across resolutions for image synthesis, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.03350)
347. Elucidating the Exposure Bias in Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2308.15321)
348. PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.00426)
349. Inner Classifier-Free Guidance and Its Taylor Expansion for Diffusion Models, ICLR 24 [[Paper]](https://openreview.net/pdf?id=0QAzIMq32X)
350. Denoising Diffusion via Image-Based Rendering, ICLR 24 [[Paper]](https://arxiv.org/pdf/2402.03445)
351. Solving Diffusion ODEs with Optimal Boundary Conditions for Better Image Super-Resolution, ICLR 24 [[Paper]](https://arxiv.org/pdf/2305.15357)
352. EfficientDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.03270)
353. Soft Mixture Denoising: Beyond the Expressive Bottleneck of Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2309.14068)
354. A Variational Perspective on Solving Inverse Problems with Diffusion Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2305.04391)
355. Denoising Diffusion Step-aware Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2310.03337)
356. Diffusion Posterior Sampling for Linear Inverse Problem Solving: A Filtering Perspective, ICLR 24 [[Paper]](https://openreview.net/pdf?id=tplXNcHZs1)
357. Exploring Diffusion Time-steps for Unsupervised Representation Learning, ICLR 24 [[Paper]](https://arxiv.org/pdf/2401.11430)
358. A Unified Sampling Framework for Solver Searching of Diffusion Probabilistic Models, ICLR 24 [[Paper]](https://arxiv.org/pdf/2312.07243)
359. Diffusion Posterior Sampling for Linear Inverse Problem Solving: A Filtering Perspective, ICLR 24 [[Paper]](https://openreview.net/pdf?id=tplXNcHZs1)
360. Imitating Human Behaviour with Diffusion Models, ICLR 23 [[Paper]](https://arxiv.org/pdf/2301.10677)
361. Accelerating Score-Based Generative Models with Preconditioned Diffusion Sampling, ECCV 22 [[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830001.pdf)
362. Unleashing Transformers: Parallel Token Prediction with Discrete Absorbing Diffusion for Fast High-Resolution Image Generation from Vector-Quantized Codes, ECCV 22 [[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830171.pdf)
363. Learning to Efficiently Sample from Diffusion Probabilistic Models, arXiv [[Paper]](https://arxiv.org/pdf/2106.03802)
364. Subspace Diffusion Generative Models, ECCV 22 [[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830274.pdf)
365. DiffuseMorph: Unsupervised Deformable Image Registration Using Diffusion Model, ECCV 22 [[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910336.pdf)
366. DiffuStereo: High Quality Human Reconstruction via Diffusion-Based Stereo Using Sparse Cameras, ECCV 22 [[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920697.pdf)
367. DiTFastAttn: Attention Compression for Diffusion Transformer Models, arXiv [[Paper]](DiTFastAttn: Attention Compression for Diffusion Transformer Models)
368. ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation, arXiv [[Paper]](https://arxiv.org/pdf/2406.02540)
369. Inf-DiT: Upsampling Any-Resolution Image with Memory-Efficient Diffusion Transformer, arXiv [[Paper]](https://arxiv.org/pdf/2405.04312)
370. PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models, arXiv [[Paper]](https://arxiv.org/pdf/2405.14430)
371. 