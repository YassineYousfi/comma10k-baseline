# 🚗 comma10k-baseline 

A semantic segmentation baseline using [@comma.ai](https://github.com/commaai)'s [comma10k dataset](https://github.com/commaai/comma10k).

Using U-Net with efficientnet encoder, this baseline reaches 0.045 validation loss.

## Visualize
Here is an example (randomly from the validation set, no cherry picking)
#### Ground truth 
![Ground truth](example.png)
#### Predicted
![Prediction](example_pred.png)


## How to use
This baseline uses two stages (i) 448x576 (ii) 896x1184 (close to full resolution)
```
python3 train_lit_model.py --backbone efficientnet-b4 --version first-stage --gpus 2 --batch-size 28 --epochs 100 --height 448 --width 576
python3 train_lit_model.py --backbone efficientnet-b4 --version second-stage --gpus 2 --batch-size 7 --learning-rate 5e-5 --epochs 30 --height 896 --width 1184 --seed-from-checkpoint .../efficientnet-b4/first-stage/checkpoints/last.ckpt
```

## WIP
- Update to pytorch lightning 1.0
- Use A.PadIfNeeded in the second stage instead of Resize
- Try more image augmentations


## Dependecies
Python 3.5+, pytorch 1.6+ and dependencies listed in requirements.txt.