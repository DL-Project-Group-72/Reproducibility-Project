# Deep Learning reproducibility project: putting *Restoring Extremely Dark Images in Real Time* to the test

For this reproducibility project, our objective is to test the model proposed in ***Restoring Extremely Dark Images in Real Time*** [1] varying both the training and the test data. In this blog post, a brief breakdown of the paper, our objectives, experiments and results are presented.

1. [Brief paper breakdown](#brief-paper-breakdown)
2. [Main project goals](#main-project-goals)
3. [Dark image datasets](#dark-image-datasets)
4. [Training the model](#training-the-model)
5. [Conclusions](#conclusions)

## Brief paper breakdown

In the aforementioned paper, a "*deep learning architecture for extreme low-light single image restoration*" [1] is presented. The main focus of this model is not only getting high-quality and "*visually appealing*" results, but also computational speed and memory efficiency. According to the authors, the proposed model's results are on the same level as the results from state-of-the-art, computationally more demanding models. The model is fast, indeed, since 4K images can be restored at 1 FPS (CPU) and at 32 FPS (GPU), allowing even for real-time video processing.

Another highlight from the original paper which is of great importance for us (and our objectives, which are depicted in the next section) is that, without any fine-tuning, the model is supposed to "*generalize well to cameras not seen during training and to subsequent tasks such as object detection*".

## Main project goals
Our principal objective is to analyze how the model performs on images from different sources (e.g.: common datasets in the literature, and even our own). We are of the opinion that this is a major aspect to look at since the test data source can potentially affect model performance [2]. This *could* lead to the model working significantly better on its own test data than on real-life test data. One reason for this could be that, even if the test data was not used at all for training, it might come from the same data distribution/source. For instance, if we take a lot of similar pictures in similar light conditions with the same camera and split them into training and test partitions, and then feed the model with different images, the latter will probably produce much worse results because they are significantly different. Whereas this extreme case doesn't apply to the original dataset used to train this model (e.g.: there are images from different locations), we want to ensure that there isn't any non-trivial bias and the generalization is indeed correct.

We also aim to explore the bounds when it comes to the number of training samples (i.e.: study model performance with increasingly large subsets of the original training set). This way, we will be able to determine the optimal number of training instances by taking into account both the model performance and the resources needed for training (computational resources such as CPU/GPU time and necessary memory, training time, and such). As far as we are concerned, this is an interesting aspect to look at since one of the aims of the paper is to reduce model complexity and computational power with respect to similar approaches.

## Dark image datasets
The first goal of our project is to try different RAW datasets of dark images on our model. In the paper, it is mentioned that "Besides being fast, our method generalizes better than the state-of-the-art in restoring images captured from a camera not seen during training and for the task of object detection." [1] This claim we obviously wanted to research, so we went looking for other datasets that could be used on the trained model.

Our search for different datasets was done via Google and by reading other papers which relate to dark imagery. From this search, there were multiple potential datasets that we could consider. First, the most commonly used dataset for dark images in RAW format is the [See-in-the-dark (SID)](https://github.com/cchen156/Learning-to-See-in-the-Dark#dataset) dataset [3]. The paper that we discuss in this blog post uses this dataset for example. Second, there is the massive [Matching in the Dark (MID)](https://github.com/Wenzhengchina/Matching-in-the-Dark) dataset [4]. Third, the [Seeing Motion in the Dark (SMID)](https://github.com/cchen156/Seeing-Motion-in-the-Dark#original-sensor-raw-data) dataset [5], which is also very large. Furthermore, we also came across the [Exclusively-Dark-Image](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset) dataset [6], which had potential, but unfortunately only turned out to contain [PNG and JPG files and no RAW files](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/issues/7)... Lastly, our group had access to a Sony A6400 camera and we were therefore able to capture RAW dark images ourselves to use as a dataset to test the performance of the model.

Since RAW images have a very large file size, these datasets are in the hundreds of gigabytes to download. Overall, the downloaded datasets took up 732 GB of storage on our computers and took more than 1 day to download. We selected a number of images from each of them to form a new smaller dataset (~4GB), which is [available using this link](https://drive.google.com/file/d/1BlQQKEfL3dBvkgP4m5nNncZYj8drp7GJ/view?usp=sharing) (externally hosted since *GitHub* isn't the best alternative for big binary files).

### Setup
To test the model on different datasets, first clone the [GitHub repository](https://github.com/MohitLamba94/Restoring-Extremely-Dark-Images-In-Real-Time) related to the paper. Please follow the `README.md` file from this repository as it provides information on the required packages. When all is set up, delete the template ARW images from the `demo_imgs` directory, but make sure to keep the `weights` file. Now, upload all dark RAW files that need to be restored in this `demo_imgs` directory. Next, navigate to the root directory and run the command `python demo.py`. All dark RAW files should get restored images inside the `demo_restored_images` directory.

### See-in-the-Dark (SID)
The SID dataset is currently the standard dataset that gets used by researchers when needing dark images in RAW format. The dataset simply consists of short exposure photographs (dark) and their long exposure counterparts (bright). The paper that we discuss in this blog post also used this dataset for its training and testing results.

In Figure 1, you can see 5 dark images that were sampled from the SID dataset and used as input in the model. The different `m` values relate to the artificial exposure that the model adds to brighten the image.
![Figure 1](https://i.imgur.com/sxWD0nS.jpg)
*Figure 1: See-in-the-Dark dataset results comparison*

Based on these image comparisons, we'd say that the model performs a very good job on restoring very dark images. The resulting images come quite close to the ground truth and objects can indeed be seen rather clearly.

### Matching in the dark (MID)
Next is the MID dataset. This dataset is really good since there are multiple configurations available for each image. Each image is available in 8 ISO levels and within each level there are 6 photos taken with different shutter speeds. Therefore there are 48 RAW files available for one and the same image. This does, however, make this dataset extremely large, namely 460 GB. Hence, downloading the complete dataset will take a long time. And even though we had a free terabyte on a hard drive, the size of the dataset was deemed too large, so we chose a selection of images.

Something to notice is the fact that this is the only dataset that is not shot on a Sony camera. All images come from a Canon camera and therefore it is quite interesting to see how well the model will generalize on this data.

![Figure 2](https://i.imgur.com/tJ9MUiw.jpg)
*Figure 2: 400+ ISO artifacts*

As can be seen in Figure 2, all images with an ISO that was equal or larger than 400 had a strange artifact to their restorations. For `m=1` there is a brown overlay, for `m=5` there is a purple overlay and `m=8` generated a blue overlay. Only the higher ISO images did show the objects a bit better, but they are still far off from the ground truth.

![Figure 3](https://i.imgur.com/lojoO5D.jpg)
*Figure 3: Matching in the dark dataset results comparison*

Based on the observation of the purple/blue hue artifact, only images of ISO 100 or 200 were chosen. In Figure 3, there is a comparison between the images, where primarily the shutter speed now plays a key role. Looking at the restored images, apparently those with a fast shutter speed seem to perform best on the restoration. This is quite remarkable as those images are actually darker than those with a slower shutter speed and should therefore, by common sense, be more difficult to restore.

### Seeing Motion in the dark (SMID)

The SMID dataset contains loads of duplicate RAW images which makes this dataset also extremely large. The dataset can be downloaded in 5 parts related to short exposure (dark) images and 1 part long exposure (bright) images. We decided to split this dataset over 2 people, since it was too large to completely download for one single person's hard drive. To give an indication of the size of this dataset: 3 parts dark images and 1 part bright images totaled 325 GB.

Figure 4 shows the restoration of 5 sampled dark images in this dataset and we were actually surprised at how poorly the model handled the restorations.
![Figure 4](https://i.imgur.com/2p5oNZh.jpg)
*Figure 4: Seeing Motion in the dark dataset results comparison*

The photographs were also created on a Sony camera and seemed to be similar to the SID dataset due to only containing short exposure photographs (dark) and their long exposure counterparts (bright). It, however, restored all dark images with a brownish overlay and the scene itself is still barely even visible with `m=8`. We had expected much better results.

### Group 72 dark images

The last dataset that we tested on the model were RAW dark images that we'd captured ourselves. We as group 72 had access to a Sony A6400 camera and decided it would be nice to try to see the actual generalizability of the model by testing it on our own photographs.

As can be seen in Figure 5, the images that we captured ourselves have quite poor exposure for the ground truth since they were captured at night. However, we were blown away by the restorations of the model. Some images came out even brighter and clearer than the available ground truth images.

![Figure 5](https://i.imgur.com/o1KqytI.jpg)
*Figure 5: Group 72 dataset results comparison*

One noticeable artifact is the color change of the keyboard. The blue lights have turned into a red/orange color, which is not similar at all to the original scene. This seems to be a bug in the model as we have spoken to other groups that were working on the reproducibility of the same paper and they have noticed the same issue. We speculate that it is related to the model architecture and that it could possibly be fixed by changing this. Nevertheless, the objects are still clearly visible.

### Smaller combined dataset
Because downloading all images from the multiple datasets takes up a lot of storage and time, we've decided to create a dataset of our own which is much smaller in size, namely 3.6 GB zipped. It contains several hand-picked images of each of the datasets. These images can then be used to test the model yourself. 

Please download our smaller combined dataset via **[Google Drive](https://drive.google.com/file/d/1BlQQKEfL3dBvkgP4m5nNncZYj8drp7GJ/view?usp=sharing)** *(possibly only available for the duration of the CS4240 Deep Learning course. If you are interested and it is not available anymore, please request access by sending an email to Mark Bekooy)*

## Training the model
Our other goal in this reproducibility project was to inspect how the model would behave given **less data**. Perhaps, a model using a smaller chunk of training data would result in the same performance. To explore this, we made use of the Google Compute Engine with the following specs:

```
vCPUs: 16
GPUs: 1 x NVIDIA Tesla T4
Memory: 104 GiB
Storage: 200 GB
```

We decided to train the model four times - with the randomly sampled training data comprising 10%, 40%, 70% and 100% of the original training data. To compare results, we recorded the structural similarity index measure (SSIM) and Peak signal-to-noise ratio (PSNR) after every 10.000 steps. These metrics were already used in the paper and allow to compare the quality of restoration not only between the variants of the model, but also with other dark image restoration approaches (such as DCE [7] & LDC [8]). Additionally, we set the number of iterations to 100.000 instead of the original 1.000.000 iterations from the paper to fit in the allocated Google Cloud budget.

![Figure 6](https://i.imgur.com/qyjcVMX.jpg)
![Figure 7](https://i.imgur.com/O3s0eS0.jpg)

In the figures above we show how the metrics were changing over time and compare them with LDC, DCE and the original paper's model. Importantly, we took the metrics of DCE, LDC and the original model from the paper. So, we recorded all other metrics by ourselves. From these two figures, two important observations can be made:

1. The model trained with 100% of the data attains the same results as in the paper, even though we only have ten times fewer iterations for training.
2. There is an expected increase in metrics when more training data gets used. Surprisingly, the model trained with only 70% of the data manages to attain similar metrics to a 100%-of-the-data model at several timesteps.

To complement the aforementioned metrics, we also present images achieved at each timestep for each of the 4 variants of the model. As can be seen, 70% and 100% model images indeed look better than the ones achieved by using less training data. Notably, all images preserve the same visual artifact - the figurine does not become blue - it has a mix of blue and brown colors in each restoration. That means that the artifact is very likely not relevant to the training data - perhaps changing the architecture of the original network might fix this interesting issue. This remains a prospect for future research.

| Ground Truth | Result from the unchanged model |
|:-------------------------:|:-------------------------:|
| ![](https://i.imgur.com/hdrx3ZX.jpg) | ![](https://i.imgur.com/fgUQRL3.jpg) |
| **10% of the data used** | **40% of the data used** |
|![](https://i.imgur.com/91ZpIpB.gif) | ![](https://i.imgur.com/irupNce.gif) |
| **70% of the data used** | **100% of the data used** |
| ![](https://i.imgur.com/qloIwda.gif) | ![](https://i.imgur.com/QoozixV.gif) |


## Final Remarks

The "Restoring Extremely Dark Images in Real Time" paper [1] mentions that the model for restoring images is able to generalize much better than the state-of-the-art in restoring images captured from a camera not seen during training. After testing the provided model on four different datasets, it appears that this statement does not completely hold. Two of the datasets show quite poor restorations of the dark images that were used as input to the model. Nevertheless, it is still very impressive how much detail can be recovered by the model from such dark images.

In general, the restoration quality is promising, specially taking into account the computational efficiency. However, the results for certain images were deceiving and other state-of-the-art models achieve better outputs. Specifically, we found two strange artifacts:
- Purple/blue hues on 400+ ISO images (Canon images instead of Sony)
- Blue objects in dark images turning red on restoration

It is also worth to mention that the experiments could be reproduced, since the performance obtained with the original dataset are almost identical to those presented in the paper. In addition, we found that 100.000 iterations may be enough to train a model with original model's performance. However, further research is needed.

## Authors

This project was carried out for TU Delft's *Deep Learning* course (CS4240) by:

- Arsenijs Nitijevskis (A.Nitijevskis@student.tudelft.nl)
- Jorge Bruned Alamán (J.BrunedAlaman@student.tudelft.nl)
- Mark Bekooy (M.J.Bekooy@student.tudelft.nl)
- Boriss Bermans (B.Bermans@student.tudelft.nl)

## References
[1] M. Lamba and K. Mitra, "Restoring Extremely Dark Images in Real Time", 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 3486-3496, doi: 10.1109/CVPR46437.2021.00349.  
[2] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, Vaishaal Shankar:
Do ImageNet Classifiers Generalize to ImageNet? CoRR abs/1902.10811 (2019)  
[3] Chen, C., Chen, Q., Xu, J., & Koltun, V. (2018). Learning to see in the dark. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3291-3300)  
[4] Song, W., Suganuma, M., Liu, X., Shimobayashi, N., Maruta, D., & Okatani, T. (2021). Matching in the Dark: A Dataset for Matching Image Pairs of Low-light Scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 6029-6038)  
[5] Chen, C., Chen, Q., Do, M. N., & Koltun, V. (2019). Seeing motion in the dark. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 3185-3194)  
[6] Loh, Y. P., & Chan, C. S. (2019). Getting to know low-light images with the exclusively dark dataset. Computer Vision and Image Understanding, 178, 30-42  
[7] Guo, C., Li, C., Guo, J., Loy, C. C., Hou, J., Kwong, S. and Cong, R, "Zero-reference deep curve estimation for low-light image enhancement", 2020, arXiv.org.  
[8] K. Xu, X. Yang, B. Yin and R. W. H. Lau, "Learning to Restore Low-Light Images via Decomposition-and-Enhancement," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 2278-2287, doi: 10.1109/CVPR42600.2020.00235  
