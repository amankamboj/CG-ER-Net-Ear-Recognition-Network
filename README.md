# CG-ER-Net-Ear-Recognition-Network

Recently biometric systems have shown improved capabilities because of the remarkable success of deep learning in solving various computer vision tasks. In ear recognition, the use of deep learning techniques is seldom due to training data scarcity. The existing work has shown poor performance as the majority of techniques are based on either handcraft features or pre-trained models. Besides this, transfer-learning has also shown poor performance because of the diversity among the tasks. To circumvent the existing issues, in this work, we have presented an end-to-end framework for ear recognition. It consist of the Ear Mask Extraction (EME) network to segment the ear, a normalization algorithm to align the ear, and a novel siamese-based CNN (CG-ERNet) for deep ear feature learning. CG-ERNet exploits domain-specific knowledge by using Curvature Gabor filters and uses triplet loss, triplet selection, and adaptive margin for better convergence of the loss. For comparative analysis, we trained state-of-the-art deep learning models like Face-Net, VGG19, ResNet50, Inception, Exception, and Mobile-Net for ear-recognition. The performance is assessed using five well-known evaluation metrics. In the extensive experimentation, our proposed model (CG-ERNet) outperformed the deep learning models and handcrafted feature based methods on four different, publicly available, benchmark datasets. To make the results more interpretable, we employ the t-SNE visualization of learned features. Additionally, our proposed method has shown robustness to various environmental challenges like Gaussian noise, Gaussian blur, up to ± 30 degrees of rotation, and 20% of occlusion.


Ear recognition network for scarce data scenario. The framework consists of three steps: 

1) Ear Segmentation, 2) Ear Alignment, and Finally, 3) Ear Recognition 4) Contemporary deep learning models.

Also, the codes are provided for the comparison with state-of-the-art deep learning models VGG19, ResNet, Inception, Xception, MobileNet in which pre-trained weights are used during training.

Currently we have provided partial codes. However, the complete code will be provided after the acceptance notification of the paper.
