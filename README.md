# INRR3CT: CT Reconstruction from Few Planar X-Rays with Application Towards Low-Resource Radiotherapy

[Project Website](https://yransun.github.io/INRR3CT/)

This is the official pytorch implementation of the deep leraning model INRR3CT for 3D CT reconstruction from few planar X-rays. The [paper](https://link.springer.com/chapter/10.1007/978-3-031-53767-7_22) is originally published on MICCAI 2023 Deep Generative Models Workshop (DGM4MICCAI 2023). The arxiv version of the paper is available [here](https://arxiv.org/abs/2308.02100).

You can also check our latest repository here: https://github.com/yransun/INRR3CT


## Code release
A simple version that could take biplanar X-rays has been released.


## Abstract
CT scans are the standard-of-care for many clinical ailments, and are needed for treatments like external beam radiotherapy. Unfortunately, CT scanners are rare in low and mid-resource settings due to their costs. Planar X-ray radiography units, in comparison, are far more prevalent, but can only provide limited 2D observations of the 3D anatomy. In this work, we propose a method to generate CT volumes from few (<5) planar X-ray observations using a prior data distribution, and perform the first evaluation of such a reconstruction algorithm for a clinical application: radiotherapy planning. We propose a deep generative model, building on advances in neural implicit representations to synthesize volumetric CT scans from few input planar X-ray images at different angles. To focus the generation task on clinically-relevant features, our model can also leverage anatomical guidance during training (via segmentation masks). We generated 2-field opposed, palliative radiotherapy plans on thoracic CTs reconstructed by our method, and found that isocenter radiation dose on reconstructed scans have <1% error with respect to the dose calculated on clinically acquired CTs using <4 X-ray views. In addition, our method is better than recent sparse CT reconstruction baselines in terms of standard pixel and structure-level metrics (PSNR, SSIM, Dice score) on the public LIDC lung CT dataset.


## Citing our work
If you find the paper useful in your research, please cite the paper:

      @inproceedings{sun2023ct,
        title={CT Reconstruction from Few Planar X-Rays with Application Towards Low-Resource Radiotherapy},
        author={Sun, Yiran and Netherton, Tucker and Court, Laurence and Veeraraghavan, Ashok and Balakrishnan, Guha},
        booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
        pages={225--234},
        year={2023},
        organization={Springer}
      }

## Train model
Run 'train.py'. Our code is designed for Xray with dimension of (1, 128, 128), CT with dimension of (1, 128, 128, 128). Please customize your dataloader in 'train.py' file and change the parameter settings of NNs accordingly.

## Acknowledgement
This work was supported by NSF CAREER: IIS-1652633.

The public datasets were used in this paper LIDC-IDRI and LUNA 16 are under Creative Commons Attribution 3.0 Unported License and Creative Commons Attribution 4.0 International License.

MONAI and clinical level pre-trained nn-UNet from MD Anderson are used during evaluation stage.
