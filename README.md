# RDFNet
This is the implementation of the models and code for the "RDFNet:RGB-D Multi-level Residual Feature Fusion for Indoor Semantic Segmentation", ICCV2017.

# File description 
- caffe-master: caffe used in our experiments
- test.py: demo code  
- Each of NYU-50 / NYU-101 / NYU-152 directory includes RDF model and its prototxt corresponding to different number of resnet layers. (*You need to change the 'nyud_dir' parameter in the prototxt.)
- data: test data  
- nyud_layers.py: input python layer
- gupta-utils-HHA: HHA generation utils by Gupta et al. [2]

# Usage
- Install Opencv 
- Compile pycaffe: modify the "Makefile.config" in caffe-master for your environment.
- Run test.py 
  - Change 'caffe_root'
  - Set the 'scale' and 'model' to test.
  - To achieve the same accuracy reported in our paper, you need to implement multi-scale (0.6~1.2) ensemble as described in the paper.
  
# Environment
Our experiments are mainly performed on Ubuntu 14.04 with CUDA7.0 / CUDNNv4 / Titan X (maxwell) / Opencv2.7
  
# Note
- Similarly to RefineNet,
  - Our implementation uses bicubic resize function for feature map resizing.
  - We remove white boundaries of the images in NYUDv2.
- Any comment for improvement is welcome as the code is not fully optimized. but please note that further maintenance will be infrequently performed.
- OOM may occur for RDF-152 with the image scale larger than 1.0 on different environtment (e.g., Titan Xp, CUDA 8.0, CUDNN v6)

# Citation
- We would like to thank Guosheng Lin [3] for invaluable help.

[1] @InProceedings{Park_2017_ICCV,
author = {Park, Seong-Jin and Hong, Ki-Sang and Lee, Seungyong},
title = {RDFNet: RGB-D Multi-Level Residual Feature Fusion for Indoor Semantic Segmentation},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}

[2] @incollection{guptaECCV14,
  author = {Saurabh Gupta and Ross Girshick and Pablo Arbelaez and Jitendra Malik},
  title = {Learning Rich Features from {RGB-D} Images for Object Detection and Segmentation},
  booktitle = ECCV,
  year = {2014},
}

[3] @article{lin2016refinenet,
  title={Refinenet: Multi-path refinement networks with identity mappings for high-resolution semantic segmentation},
  author={Lin, Guosheng and Milan, Anton and Shen, Chunhua and Reid, Ian},
  journal={arXiv preprint arXiv:1611.06612},
  year={2016}
}

# License
For academic usage, the code is released under the permissive BSD license. For any commercial purpose, please contact the authors.
