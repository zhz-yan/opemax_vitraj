# opemax_vitraj

This is the code for "Performance Evaluation of Appliance Identification Methods under Open-Set Conditions in Non-Intrusive Load Monitoring".

**It is worth mentioning that, in our practical tests, the results of the OpenMax method were, in some cases, even worse than those of SoftMax Thresholding. This finding aims to argue that in NILM tasks, some image-based feature methods, such as the HSV VI features studied in this paper, may achieve promising results in closed-set conditions. However, they tend to fail in open-set testing, where unknown loads are introduced. Even with the introduction of well-known OSR techniques from the computer vision field, no improvement was observed. This conclusion suggests a need to rethink the OSR problem within NILM.**

## Requirements
```bash
pip install torch torchvision
pip install scikit-learn
pip install numpy
pip install libmr

```
## Data
The training and testing data can be found in 'data/dataset-name/hsv_vi.npy' and 'data/dataset-name/labels.npy'.

To see the RGB image of the HSV V-I trajectory, please see 'notebook/plot_hsv_vi_traj.ipynb'.

## Training

This is a PyTorch implementation of the project, run following command to train inital model, test will run after training ends:

#### Single Unknown Appliance Tests
```bash
python main.py --dataset plaid --u_class '0'
```
#### Multiple Unknown Appliance Tests
```bash
python main.py --dataset cooll --u_class '0_4_1'
```
## Results

The trained model state dict and results will be saved under 'checkpoints/dataset-name/unknown_class/'. We have uploaded some results for review.

To check the results, please see 'check_results.py'.

## Acknowledgement
During the implementation we base our code mostly on the [PLAID](https://github.com/jingkungao/PLAID) by Jingkun Gao, we are also inspired by the [Open-Set-Recognition](https://github.com/ma-xu/Open-Set-Recognition) implementation by Xu Ma and [MLCFCD](https://github.com/sambaiga/MLCFCD) from Anthony Faustine. Many thanks to these authors for their great work!

