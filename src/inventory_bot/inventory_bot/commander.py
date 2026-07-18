#!/usr/bin/env python3
import math
import os
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import onnxruntime as ort
class ImageSnapper(Node):
    def __init__(self):
        super().__init__('image_snapper')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        #Subscribe to topic where the camera feeds are published
        self.subscription = self.create_subscription(Image,'/camera/image_raw',self.image_callback,qos_profile_sensor_data)    
        self.bridge =CvBridge()
        self.save_path =""
        self.capture_requested =False
        self.image_saved =False

    def trigger_capture(self, path):
        self.save_path =path
        self.image_saved =False
        self.capture_requested =True
        self.get_logger().info(f"Waiting for image at: {path}")

    def image_callback(self, msg):
        #Gets called whenever a camera frame is published
        if self.capture_requested and not self.image_saved:
            try:
             cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
             cv2.imwrite(self.save_path, cv_image)
             self.get_logger().info(f"Photo saved at {self.save_path}")
             self.image_saved =True
             self.capture_requested =False
            except:
                self.get_logger().error(f"Failed to save image")

def build_pose(navigator, x, y, heading_degrees):
    pose =PoseStamped()
    pose.header.frame_id ="map"
    pose.header.stamp =navigator.get_clock().now().to_msg()
    pose.pose.position.x =x
    pose.pose.position.y =y
    pose.pose.position.z =0.0
    yaw =math.radians(heading_degrees)
    pose.pose.orientation.x =0.0
    pose.pose.orientation.y =0.0
    pose.pose.orientation.z =math.sin(yaw/2.0)
    pose.pose.orientation.w =math.cos(yaw/2.0)
    return pose

def format_yolov8(frame):
    #Format the camera image so that it does not get stretched or squeezed when converted to size 640x640
    row,col,_ =frame.shape
    maximum =max(col, row)
    result =np.zeros((maximum, maximum, 3), np.uint8)
    result[0:row, 0:col] =frame
    return result

def run_inference(image_path, output_path, session, input_name):
    #Run the images through the model for prediction
    CLASSES =['black wall', 'blue wall', 'cube', 'cylinder', 'green wall', 'red object', 'sphere', 'yellow wall']
    CLASS_CUBE =2
    CLASS_CYLINDER =3
    CLASS_RED_OBJECT =5
    CLASS_SPHERE =6
    #IOU_THRESHOLD of 0.4 causes two bounding boxes that overlap by more than 40 percent to be considered as a single class
    IOU_THRESHOLD =0.4 
    RED_CONFIDENCE_THRESHOLD =0.60 #Threshold confidence for the 'red object' class
    if not os.path.exists(image_path):
     print(f"[ERROR]There is no image at {image_path}")
     return
    print(f"Predicting on: {os.path.basename(image_path)}")
    original_image =cv2.imread(image_path)
    image =format_yolov8(original_image)
    x_factor =image.shape[1]/640.0
    y_factor =image.shape[0]/640.0
    blob =cv2.dnn.blobFromImage(image, 1/255.0, (640,640), swapRB=True, crop=False)
    outputs =session.run(None, {input_name:blob})
    preds =outputs[0][0] 
    preds =np.transpose(preds)
    boxes =[]
    confidences =[] #Holds absolute confidence values
    class_ids =[]
    normalized_shape_confidences =[] #Holds relative confidence values for classes 'cube','cylinder','sphere' relative to each other
                                      #Check readme for more information
    for row in preds:
        red_confidence =float(row[4 + CLASS_RED_OBJECT])
        if red_confidence >RED_CONFIDENCE_THRESHOLD:
            shape_probs ={
            CLASS_CUBE:float(row[4 + CLASS_CUBE]),
            CLASS_CYLINDER:float(row[4 + CLASS_CYLINDER]),
            CLASS_SPHERE: float(row[4 + CLASS_SPHERE])
            }
            sum_of_shape_probs =sum(shape_probs.values())
            if sum_of_shape_probs >0:
                chosen_shape_id =max(shape_probs, key=shape_probs.get)
                raw_shape_conf =shape_probs[chosen_shape_id]
                normalized_conf =raw_shape_conf/sum_of_shape_probs
            else:
                chosen_shape_id =CLASS_SPHERE 
                normalized_conf =0.0

            x_center,y_center,w,h =row[0:4]
            x_center*=x_factor
            y_center*=y_factor
            w*=x_factor
            h*=y_factor
            x = int(x_center-(w/2))
            y = int(y_center-(h/2))
            boxes.append([x,y,int(w),int(h)])
            confidences.append(red_confidence)
            class_ids.append(chosen_shape_id)
            normalized_shape_confidences.append(normalized_conf)

    indices = cv2.dnn.NMSBoxes(boxes,confidences,RED_CONFIDENCE_THRESHOLD,IOU_THRESHOLD)
    
    if len(indices) >0:
        for i in indices.flatten():
            x,y,w,h =boxes[i]
            class_id=class_ids[i]
            rc=confidences[i]
            n_sc=normalized_shape_confidences[i]
            label=CLASSES[class_id]
            print(f"Detected: {label.upper()} (Red Conf: {rc:.2f} | Relative Shape Conf: {n_sc:.2f})")
            color =(0,255,0)
            cv2.rectangle(original_image,(x,y),(x+w,y+h),color,2)
            display_text=f"{label} (SC: {n_sc:.2f})"
            cv2.putText(original_image, display_text, (x, max(10, y - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        print("No red objects detected!")
        
    cv2.imwrite(output_path,original_image)
    print(f"Saved image with bounding box to:{output_path}")

def main():
    rclpy.init()
    snapper=ImageSnapper()
    navigator= BasicNavigator()
    #EDIT THE WORKSPACE NAME TO WHATEVER YOURS IS
    base_dir =os.path.expanduser("~/task4_ws/src/inventory_bot/my_cnn") #~/<workspace_name>/src/inventory_bot/my_cnn
    model_path = os.path.join(base_dir,"best.onnx")
    os.makedirs(base_dir, exist_ok=True)
    if not os.path.exists(model_path):
        print(f"[ERROR] Could not find model at {model_path}")
        return    
    print("Loading model session")
    session =ort.InferenceSession(model_path)
    input_name =session.get_inputs()[0].name
    try:
        navigator.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        print("Waiting for nav2 stack to become active...")
        navigator.waitUntilNav2Active()
        print("Routing")
        waypoints = [
            ("ROOM-A", build_pose(navigator,-3.41,-2.42,5)),
            ("ROOM-B", build_pose(navigator,-1.8,1.8,90)),
            ("ROOM-C", build_pose(navigator,-4.95,2.1,90))
        ]
        #Spin the snapper briefly to let DDS discover the bridge
        for _ in range(10):
            rclpy.spin_once(snapper,timeout_sec=0.1)

        for station_name, target_pose in waypoints:
            print(f"\n[DESPATCH] Routing to: {station_name}")
            navigator.goToPose(target_pose)
            while not navigator.isTaskComplete():
                rclpy.spin_once(snapper, timeout_sec=0.01)
                feedback =navigator.getFeedback()
                if feedback:
                    dist =feedback.distance_remaining
                    eta =feedback.estimated_time_remaining.sec
                    print(f"-> Distance: {dist:.2f}m | ETA: {eta}s", end="\r")        
            result =navigator.getResult()
            if result==TaskResult.SUCCEEDED:
                print(f"\n[SUCCESS] Arrived at: {station_name}")
                raw_image_path =os.path.join(base_dir, f"{station_name}_snapshot.jpg")
                pred_image_path =os.path.join(base_dir, f"predicted_{station_name}_snapshot.jpg")
                print(f"Taking picture for {station_name}")
                snapper.trigger_capture(raw_image_path)
                timeout_counter =0
                max_attempts =2000
                while rclpy.ok() and not snapper.image_saved and timeout_counter < max_attempts:
                    rclpy.spin_once(snapper, timeout_sec=0.1)
                    timeout_counter +=1

                if snapper.image_saved:
                    print(f"Success! Image saved.")
                    #Pause exactly 0.5 seconds before prediction
                    time.sleep(0.5)
                    #Run the inference on the pic we just took
                    run_inference(raw_image_path, pred_image_path, session, input_name)
                else:
                    print(f"[ERROR] Timed out waiting for image. Moving on anyway!")
                
            elif result ==TaskResult.CANCELED:
                print(f"\n[ERROR]Task cancelled for: {station_name}")
                continue 
            elif result ==TaskResult.FAILED:
                print(f"\n[ERROR] Navigation failed for: {station_name}")
                continue
    except KeyboardInterrupt:
        print("\n[MISSION ABORTED] Script interrupted by user.")      
    finally:
        print("\n[MISSION COMPLETE] All waypoints executed.")
        snapper.destroy_node()
        navigator.destroy_node()
        rclpy.shutdown()

if __name__=="__main__":
    main()