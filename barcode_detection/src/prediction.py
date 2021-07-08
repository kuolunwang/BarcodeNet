#!/usr/bin/env python3

import numpy as np
import cv2
import os
import gdown
import rospy
import rospkg
from cv_bridge import CvBridge, CvBridgeError
import torch
import torch.nn as nn
from model.BarcodeNet import BarcodeNet
from sensor_msgs.msg import Image
from barcode_detection.srv import GetPrediction, GetPredictionResponse


class Prediction:
    def __init__(self):
        self.cv_bridge = CvBridge()
        r = rospkg.RosPack()
        path = os.path.join(r.get_path('barcode_detection'), "weight")
            
        self.labels = ['background', 'barcode']
        self.net = BarcodeNet(len(self.labels))

        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.net = self.net.cuda()

        if not os.path.exists(path):
            os.makedirs(path)

        model_path = self.__download_model(path)

        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()

        self.image_pub = rospy.Publisher("~predict_img", Image, queue_size = 1)
        self.mask_pub = rospy.Publisher("~mask", Image, queue_size = 1)
        self.img_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.predict_cb)

        rospy.loginfo('barcode predict node ready!')

    def __download_model(self, path):
        model_url = 'https://drive.google.com/uc?export=download&id=1gu71PIYOfLR1J3Dq8YIvV-m-1OLDrPHh'
        model_name = 'barcodenet'

        if not os.path.exists(os.path.join(path, model_name + '.pkl')):
            gdown.download(model_url, output=os.path.join(path, model_name + '.pkl'), quiet=False)
    
        print("Finished downloading model.")
        return os.path.join(path, model_name + '.pkl')

    def predict_cb(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        predict = self.predict(cv_image)

        _ = self.confirm_barcode(cv_image, predict)
        
        self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        self.mask_pub.publish(self.cv_bridge.cv2_to_imgmsg(predict, "8UC1"))

    def predict(self, img):
        means = np.array([103.939, 116.779, 123.68]) / 255.
        img = img / 255.
        img[:, :, 0] -= means[0]
        img[:, :, 1] -= means[1]
        img[:, :, 2] -= means[2]

        x = torch.from_numpy(img).float().permute(2, 0, 1)
        x = x.unsqueeze(0)
        if self.use_gpu:
            x = x.cuda()
        output = self.network(x)
        output = output.data.cpu().numpy()
        _, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, len(self.labels)).argmax(axis=1).reshape(1, h, w)
        pred = pred[0]
        pred = np.int8(pred)

        pred[pred != 0 ] = 255
        pred = pred.astype(np.uint8)

        return pred

    def confirm_barcode(self, img ,mask, threshold=0):

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img,contours,-1,(0,0,255),1) 
        
        area_list = []
        for i in range(len(contours)):
                
            area = cv2.contourArea(contours[i])
            area_list.append(area)

        barcode_area = np.max(area_list) if len(area_list) != 0 else 0
        if barcode_area <= threshold:
            return False
        else:     
            return True

    def onShutdown(self):
        rospy.loginfo("Shutdown.")

if __name__ == '__main__':
    rospy.init_node('barcode_predict_node', anonymous=False)
    prediction = Prediction()
    rospy.on_shutdown(prediction.onShutdown)
    rospy.spin()