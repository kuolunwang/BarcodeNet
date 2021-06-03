# BarcodeNet

This is ros package for BarcodeNet, you need to clone this repo for your workspace and re-catkin make, then you can type below command to use it.\
**The alogrithm only run in GPU computer!**

## Start use repo
```
     $ cd [your workspace]/src && git clone --recursive git@github.com:kuolunwang/BarcodeNet.git
     $ cd BarcodeNet && source install.sh
     $ source BarcodeNet
     $ cd [your workspace] && catkin make
```

## Start predict by BarcodeNet
```
     $ roslaunch barcode_detect barcode_predict.launch
```