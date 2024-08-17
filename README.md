# DCFormer
DCFormer: Divide-and-Conquer in 3D Human Pose Estimation Tasks

Thank you for your interest, the code and checkpoints are being updated.

## The released codes include
    checkpoint/:                        the folder for model weights of DCFormer.
    dataset/:                           the folder for data loader.
    common/:                            the folder for basic functions.
    model/:                             the folder for DCFormer network.
    run_staf.py:                        the python code for STAF^2ormer networks training.
    run_dc_scale.py:                    the python code for Module Scale of DCFormer networks training.
    run_dc_depth.py:                    the python code for Module Depth of DCFormer networks training.
    run_dc_agg.py:                      the python code for Module Agg of DCFormer networks training and the entire DCFormer networks testing.

## Environment
Make sure you have the following dependencies installed:
* PyTorch >= 0.4.0
* NumPy
* Matplotlib=3.1.0

## Datasets
Our model is evaluated on [Human3.6M](http://vision.imar.ro/human3.6m) and [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) datasets.
### Human3.6M
We set up the Human3.6M dataset in the same way as [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md). 
### MPI-INF-3DHP
We set up the MPI-INF-3DHP dataset in the same way as [P-STMO](https://github.com/paTRICK-swk/P-STMO). 

## Evaluation
You can download our pre-trained models from [Google Drive]() or [Baidu Disk]() (extraction code：1234). Put them in the ./checkpoint directory.
### Human3.6M
To evaluate our DCFormer model on the 2D keypoints obtained by CPN, please run:
```bash
 python run_dc_agg.py -f 243 -b 128  --train 0 --layers 6 -s 1 -k 'cpn_ft_h36m_dbb' --reload 1 --previous_dir_scale ./checkpoint/model_243_DCFormer/agg_scale_1_4030.pth --previous_dir_depth ./checkpoint/model_243_DCFormer/agg_depth_1_4030.pth --previous_dir_agg ./checkpoint/model_243_DCFormer/agg_1_4030.pth
```
Different models use different configurations as follows.

If you want to evaluate traditional method STAF^2ormer, please run:
```bash
 python run_staf.py -f 243 -b 128 --train 0 --layers 6 -s 1 -k 'cpn_ft_h36m_dbb' --reload 1 --previous_dir ./checkpoint/model_243_DCFormer/staf_11_4082.pth
```
### MPI-INF-3DHP
The pre-trained models and codes for DCFormer are currently undergoing updates. In the meantime, you can run this code to observe the results for 81 frames.

```bash
 python run_3dhp_dc_agg.py -f 81 -b 128  --train 0 --layers 6 -s 1 --reload 1 --previous_dir_scale ./checkpoint/model_81_DCFormer/3dhp_agg_scale_4_2132.pth --previous_dir_depth ./checkpoint/model_81_DCFormer/3dhp_agg_depth_4_2132.pth --previous_dir_agg ./checkpoint/model_81_DCFormer/3dhp_agg_4_2132.pth
```
## Training from scratch
### Human3.6M
To train Module Scale of DCFormer, please run:
```bash
 python run_dc_scale.py -f 27 -b 128  --train 1 --layers 6 -s 3 
```
if you want to use spatial masks, add --MAE at the end.

To train Module Depth of DCFormer, please run:
```bash
 python run_dc_depth.py -f 27 -b 128  --train 1 --layers 6 -s 3 
```
The last step is to train Module Agg and test the entire model, please run：
```bash
 python run_dc_agg.py -f 27 -b 128  --train 1 --layers 6 -s 3 --previous_dir_scale ./checkpoint/your_best_scale_module.pth --previous_dir_depth ./checkpoint/your_best_depth_module.pth
```
If you want to train traditional method STAF^2ormer, please run:
```bash
 python run_staf.py -f 27 -b 128  --train 1 --layers 6 -s 3 
```

## Visulization
Accroding MHFormer, make sure to download the YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA) and put it in the './demo/lib/checkpoint' directory firstly. Then, you need to put your in-the-wild videos in the './demo/video' directory.

You can modify the 'get_pose3D' function in the 'vis_dc.py' or 'vis.py' script according to your needs, including the checkpoint and model parameters, and then execute the following command:

```bash
 python demo/vis_dc.py --video sample_video.mp4
```





