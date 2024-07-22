# opemax_vitraj

This is the code for "Performance Evaluation of Appliance Identification Methods under Open-Set Conditions in Non-Intrusive Load Monitoring".

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

The trained model state dict and results will be saved under 'checkpoints/dataset-name/unknown_class/'

## Results
To check the results, please see 'check_results.py'.


