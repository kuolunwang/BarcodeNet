# BarcodeNet

This is ros package for BarcodeNet, you need to clone this repo for your workspace and re-catkin make, then you can type below command to use it.\
**The alogrithm only run in GPU computer!**

## How to Use BarcodeNet

---

```
     cd [your workspace]/src && git clone --recursive git@github.com:kuolunwang/BarcodeNet.git
     cd BarcodeNet && source install.sh
     cd [your workspace] && catkin_make
```

## Start Predict by BarcodeNet

---

Launch this file and open Rviz to see predicted result and mask. **Please make sure you open camera before launch predict file, in this case, we use D435 for example.**
```
     roslaunch barcode_detection barcode_predict.launch
```

## Topic List

---

| Topic Name | Topic Type | Topic Description |
|:--------:|:--------:|:--------:|
| /BarcodeNet/predict_img | [Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)| This topic will show predicted result in origin img|
| /BarcodeNet/mask | [Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html) | This topic will show predicted mask with binary format |

## Related Information

---

We provided [dataset](https://drive.google.com/drive/folders/13o1_pha07T4vPZWcdGhQLQwVct_67r1d?usp=sharing) and its [pre-trained weight](https://drive.google.com/file/d/1-K8VKhCRbl6e3E26RYxtI_VY4skxTbcc/view?usp=sharing), also provided a simple inference on colab.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RpBTx4LjsyDzocN52e8B6tfqtpJfxKjc?authuser=1#scrollTo=4bDdq5kf7mBq)

### Dataset Description

---

The train and test dataset collected by Intel Realsense D435, and use Labelme to label barcode, save mask image to the folder. The training image randomly placed one or two objects, the detailed information can refer below.

* Class : 2 (barcode and background)
* image size : 640*480
* Object : 15 kinds of objects with different level of distortion

| Name | Dataset | # of images & masks |
|:--------:|:--------:|:--------:|
| Train & Eval Dataset | [train_eval_data](https://drive.google.com/file/d/1ieflBYDi4MjFMKTIsEBz0iquZEi9xagl/view?usp=sharing) | 8547 for training and 950 for evaluation split by 9:1(training/evaluation)|
| Test Dataset | [test_data](https://drive.google.com/file/d/16wQx_5KOBUVXyuumFIY6TF7xyZ4sBve0/view?usp=sharing)| 100 in one_product scene and 50 in two_products scene |

## Next to do 

1. Training code 