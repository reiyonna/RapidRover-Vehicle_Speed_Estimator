import datetime
import math
import torch
import cv2
from ultralytics import YOLO
import argparse
import utils.draw as draw
import os
from ultralytics.utils.plotting import Annotator 

# Set environment variable to allow multiple instances of OpenMP library
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("\nINFO: This program uses your hardware resources to perform machine learning based speed detection.\nFor best visual results, it is recommended to run the program on a plugged-in laptop or a desktop computer.\n")

def init_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Runs speed detection on video')
    parser.add_argument("-s", "--source", default=None, help="The source for the detector to run on, \
                              can be the path to a directory or a file.")
    parser.add_argument("--no-visual", default=False, help="Whether to run the detection in background without visual feedback.")
    parser.add_argument('-v', '--verbose', type=bool, default=False, help="Whether to print additional detected information to the stdout.")
    parser.add_argument('--hide', type=bool, default=False, help="Hide detected information on the video output.")
    args = parser.parse_args()
    return args

color_scheme = {
     "car": (0, 0, 225),
     "van": (180, 105, 255),
     "bus": (0, 140, 255),
     "others": (152, 152, 103)
}

ROOT_OUTPUT_DIR = "static/videos/"

#This program uses your hardware resources to perform machine learning based speed detection. 
#For best visual results, it is recommended to run the program on a plugged-in laptop or a desktop computer.
class SpeedDetector:
     def __init__(self, args, video_name, out_dir) -> None:
          self.args = args
          self.video_name = video_name
          self.out_dir = out_dir
          self.save_file = os.path.join(self.out_dir, video_name)
          self.frame_rate = 0
          self.speed_detections = {}
          self.ready()

     def ready(self):
          # use gpu if available else use cpu
          device = 'cuda' if torch.cuda.is_available() else 'cpu'
          torch.device(device)

          # load the trained model
          self.model = YOLO(task="detect")
          # self.model = YOLO('best.pt', task="detect")
          print(f"Running on device type: {device}")

     def detect(self):
          out_dir = os.path.join(self.out_dir, self.video_name)
          # out_dir = self.out_dir

          if not os.path.exists(out_dir):
               os.makedirs(out_dir)
          
          self.save_file = os.path.join(out_dir, f"generated_output_{os.path.basename(self.video_name)}")

          #video saver
          fourcc = cv2.VideoWriter_fourcc(*'avc1') #(*'avc1'/'MP42)
          out = cv2.VideoWriter(f"{self.save_file}.mp4", fourcc, 20.0, (1280, 960)) #avi

          vidcap = cv2.VideoCapture(self.video_name)
          frame_count = 0

          prev_speed_queue = {}
          total_detections = 0

          if args.no_visual:
               print("Please wait... running in background...")

          while vidcap.isOpened():
               success, frame = vidcap.read()
               if success:
                    # count the frame number, starting with 1
                    frame_count += 1
                    #variable naming, improve.
                    results = self.model.track(frame, persist=True, verbose=self.args.verbose)

                    for result in results:
                         annotator = Annotator(frame)
                         
                         boxes = result.boxes
                         count = len(boxes)

                         info = {"car": 0, "van": 0, "bus": 0, "others": 0}

                         for box in boxes:
                              b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                              c = box.cls
                              confidence = round(float(box.conf), 2)
                              x, y, z, v = box.xyxy[0]
                              detected_type = result.names[int(c)]
                              # --------------Added for unrecognized in info for pretrained YOLO---------------
                              if detected_type not in color_scheme:
                                   detected_type = "others"
                              # ---------------------------------------
                              if not box.id:
                                   break

                              obj_id = int(box.id) #can be none, if not available

                              info[detected_type] += 1

                              if not self.args.hide:
                                   try:
                                        self.frame_rate = round(1 / (result.speed["inference"] / 1000))
                                   except ZeroDivisionError:
                                        # if the inference time is zero or very close to zero, the FPS will be equal to the refresh rate of the monitor, 
                                        # or the maximum FPS that the monitor can support, here to avoid further imports, we default to 60.
                                        self.frame_rate = 60
                                        
                                   frame = draw.draw_text_with_background(frame, f"Total Detections: {total_detections}; FPS: {self.frame_rate}", (0, 0))
                                   frame = draw.draw_text_with_background(frame, f"Total Count: {count}", (0, 30))
                                   frame = draw.draw_text_with_background(frame, f"Cars: {info['car']}  ", (0, 60))
                                   frame = draw.draw_text_with_background(frame, f"Buses: {info['bus']}", (0, 90))
                                   frame = draw.draw_text_with_background(frame, f"Vans: {info['van']}  ", (0, 120))
                                   if info["others"]:
                                        frame = draw.draw_text_with_background(frame, f"Others: {info['others']}  ", (0, 150))

                              if obj_id not in prev_speed_queue: 
                                   # if not, add it with the current location as the value
                                   prev_speed_queue[obj_id] = (x, y) 
                                   speed = 0
                              else:
                                   # if yes, get the previous location from the value
                                   prev_x, prev_y = prev_speed_queue[obj_id] 
                                   # call the estimate_speed function with the previous and current locations as the arguments
                                   speed = self.estimate_speed((prev_x, prev_y), (x, y)) 
                                   # update the dictionary with the current location as the new value
                                   prev_speed_queue[obj_id] = (x, y) 
                                   # display the speed along with the other information on the frame

                              out_data = [
                                        f"{self.video_name} {frame_count} {obj_id} {int(x)} {int(y)} {int(z)} {int(v)} {speed} {confidence}"
                                   ]
                              
                              if obj_id in self.speed_detections:
                                   self.speed_detections[obj_id]["data"] += out_data
                              else:
                                   total_detections += 1
                                   self.speed_detections[obj_id] = {"time": datetime.datetime.now(), "data": out_data }
                                   
                              annotator.box_label(b, f"{speed} km/h id:{obj_id} {detected_type} {confidence}", color=color_scheme[detected_type])
                              
                    # display the annotated frame
                    if not args.no_visual:
                         cv2.imshow("YOLOv8 Tracking", frame)

                    out.write(frame)

                    # exit if q is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                         #clean exit?
                         self.cleanup()
                         break
               else:
                    self.cleanup()
                    print(f"Thank you! Finished detections, saved to {self.save_file}")
                    break
          else:
               raise Exception("Video could not be parsed") 
          
          # release the video capture object
          vidcap.release() 
          #out.release()
          cv2.destroyAllWindows()

     def estimate_speed(self, location1, location2):
          #Added frame rate differing, based on the distance formula and the speed formula
          framerate = 15 * 2
          d_pixel = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
          ppm = 8 #pixels per meter ratio
          d_meters = d_pixel/ppm
          time_constant = framerate
          speed = d_meters * time_constant

          return round(speed)

     def cleanup(self):
          self.create_record(self.speed_detections)

     def create_record(self, info):
          """
          Creates a record of file type .txt

          syntax of generated files:

          <test_video_name> <frame_no> <obj_id> <xmin> <ymin> <xmax> <ymax> <speed> <confidence>
          """
          output_dir = self.out_dir
          output_dir = os.path.dirname(self.save_file)
          if not os.path.exists(output_dir):
               os.makedirs(output_dir) 

          with open(os.path.join(output_dir, "speed_result.txt"), "w") as f:
               for id in info:
                    writeable_data = "\n".join(self.speed_detections[id]["data"]) + "\n"
                    f.write(writeable_data)


if __name__ == "__main__":
     print("\nINFO: This program is flags enabled! call with -h flag to see the help menu!\n") 
     args = init_argparse()

     path = args.source if args.source else "test_videos/scene1_01.mp4"

     if os.path.isdir(path):
          for file in os.listdir(path):
               file_path = f"{path}/{file}"
               print(f"Running detection on {file_path}")
               SpeedDetector(args, file_path, ROOT_OUTPUT_DIR).detect()
     elif os.path.isfile(path):
          print(f"Running detection on {path}")
          SpeedDetector(args, path, ROOT_OUTPUT_DIR).detect()
     else:
          raise Exception("Invalid path provided: the provided source is not a directory or a valid path.")
