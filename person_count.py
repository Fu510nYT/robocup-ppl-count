#!/usr/bin/env python3
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import torch
from pytorch_models import *

def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")

if __name__ == "__main__":
    rospy.init_node("Person_Countt")
    rospy.loginfo("Person Count Node")

    # Ros Messages

    _frame = None
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback_image)
    
    # Initalizing Models
    dnn_yolo = Yolov5()
    previous_x = 0
    
    people_count = 0
    
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        if _frame is None: continue
        
        frame = _frame.copy()
        
        # Visualize Lines
        cv2.line(
            frame, 
            (_frame.shape[1] // 2, 0), 
            (_frame.shape[1] // 2 , _frame.shape[0]), 
            (0, 0, 255),
            1
        )
        
        
        boxes = dnn_yolo.forward(frame)
        for _id, index, conf, x1, y1, x2, y2 in boxes:
            if dnn_yolo.labels[index] == "person":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, dnn_yolo.labels[index], (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) 
                
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                if previous_x < _frame.shape[1] // 2 and cx > _frame.shape[1] // 2:
                    people_count += 1
                if previous_x > _frame.shape[1] // 2 and cx < _frame.shape[1] // 2:
                    people_count -= 1
                
                if people_count < 0: people_count = 0                
                previous_x = cx
            cv2.putText(frame, str(people_count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 1)
        
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) in [27, ord('q')]:
            break


