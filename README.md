![cover page demonstrating speed defection](/assets/profile.png)

[Skip to execution steps](#execute)

test video set data available in **detections/speed_result.txt**

# Vehicle Speed Detection using YOLOv8 and DETRAC Dataset
This project aims to detect the speed of vehicles from a video using a deep learning model called YOLOv8.

The detection model is trained on the publicly available DETRAC dataset, which is a large-scale benchmark for vehicle detection and tracking in urban traffic scenes. The dataset contains more than 140,000 frames with more than 1.2 million annotated bounding boxes of vehicles. 

The project demonstrates the potential of deep learning and machine learning to be employed in vehicle speed detection, it can have various implementations in areas like traffic management, law enforcement and autonomous driving. This project also shows the limitations including limitations in computational cost for training models and accuracy in speed detection.

Everything from the preprocessing, training, evaluation and testing has been done from the **scratch**.

The project consists of the following steps:

#### Training
1) preprocessing the data set from DETRAC by parsing the XML file to and extracting the information and then creating relevant TXT files as required by yolo, converting the video frames by renaming to associated text file of the generated annotation file.
[see structure generated](#output)

#### Detection
* Running the trained model on each individual frame of the video source video, feeding the detected vehicles into a speed estimating algorithm.
* At the same time we keep track of relevant information such as the frame number, the vechicle ID, the bounding box coordinates, total number of vehicles, the vehicle type, and the speed.
* Use cv2 to draw the necessary information on the frame and then show the annotated and processed frame with the metadata to the user.

# <a name="execute"></a> Execution
To run the speed detection script, follow the steps provided:
1. run `pip3 install -r requirements.txt`
2. `python main.py` (if no source is passed, program with use a test video) you can pass in a video source as an argument like so
 `python main.py -s source/to/video_or_folder_containing_videos` (the path can either be a directory containing videos or a video) 
* for help with the CLI and to see further commands run `python main.py -h`

# training steps
> Note:
This step is not required unless you want to train it yourself, a pre-trained model is available.

# Preprocessing 
The first step is in converting the labels which are provided in XML format to .txt files as required by yolo
format.

In the first step, the script automatically parses the xml using beautiful soup, extracts the necessary info and then write to respective .txt file.

After that step, the image processing is done, here the images are recursively copied to the output folder.

All of this can be invoked by running a single python command for simplicity.

# Installation
#### (automated-installer)
If you running on a device running linux, we have a bash script that automates the dataset installation and preprocessing of the dataset for you!
```bash
wget https://raw.githubusercontent.com/NihalNavath/YOLOV8-train-on-detrac-dataset/main/setup.sh && chmod u+x setup.sh && ./setup.sh
```

#### (Manual installation)
1) dataset and annotation download
```bash
# download images
curl -O https://detrac-db.rit.albany.edu/Data/DETRAC-train-data.zip
curl -O https://detrac-db.rit.albany.edu/Data/DETRAC-test-data.zip

# unzip images
unzip DETRAC-train-data.zip
unzip DETRAC-test-data.zip

# download annotations
curl -O https://detrac-db.rit.albany.edu/Data/DETRAC-Train-Annotations-XML.zip
curl -O https://detrac-db.rit.albany.edu/Data/DETRAC-Test-Annotations-XML.zip

# unzip annotations

# if on windows,
# right click > extract all
unzip DETRAC-Train-Annotations-XML.zip
unzip DETRAC-Test-Annotations-XML.zip

# optional remove
rm -rf DETRAC-test-data.zip
rm -rf DETRAC-train-data.zip
rm -rf DETRAC-Train-Annotations-XML.zip
rm -rf DETRAC-Test-Annotations-XML.zip

```

2) Clone this git repo
```bash
git clone https://github.com/NihalNavath/YOLOV8-train-on-detrac-dataset.git
```
3) Navigate to cloned repo 
```bash
cd YOLOV8-train-on-detrac-dataset
```

**The next step is only required for Linux devices**
create virtual environment (**can skip on certain distros**, but recommended)
```bash
# This step is only for linux devices! Skip if you are on Windows.
python3 -m venv .venv
source .venv/bin/activate
```

4. download dependencies 
```bash
pip3 install -r requirements.txt
```

5. Run the main parser command
```bash
cd preprocessor
python parser.py
#optional cleanup
rm -rf temp
```

and voila! you should be done! In the end the script should print set of two numbers which should look like this
```
train images: 83791; train labels: 82085;
valuation images: 56340; valuation labels: 56167;
```

# Format (info)
### Annotations txt format
The generated text files follow the following format 
`class_index, x_center, y_center, w, h`
where, 
* `class_index` is the type of vehicle which can be either of the following
                            0: car
                            1: van
                            2: bus
                            3: others

* `x_center` is the x-coordinate of the center of the bounding box,
* `y_center` is the x-coordinate of the center of the bounding box,
* `w` the width
* and `h` the height

## Generated text files after detection
The generated text files will be placed in `detections/PATH_TO_VIDEO_FILE/VIDEO_NAME/speed_result.txt`
The generated video files will be placed in `detections/PATH_TO_VIDEO_FILE/VIDEO_NAME/generated_output.avi`

The text files follow this format, values are space delimited.

```
<test_video_name> <frame_no> <obj_id> <xmin> <ymin>  <xmax> <ymax>  <speed> <confidence>
```

where,
1) `<test_video_name>` is the test video file name.

2) `<frame_no>` represents the frame count for the current frame in the current video, starting with 1.

3) `<obj_id>` is a numeric identifier for the vehicle. It is an integer.

4) The xyxy coordinates of the bounding box

5) `<speed>` denotes the instantaneous speed of the vehicle in the given frame, measured in kilometer per hour (km/h), which is a non-negative real value.

6) `<confidence>` denotes the confidence of the prediction. will be between 0 and 1.

### Generated structure of output after preprocessing
#### <a name="output"></a> > main output folder
```
.
└── output /
    ├── images /
    │   ├── train
    │   └── val
    └── labels/
        ├── train
        └── val
```
#### > temporary folder
```
temp
  train
    MVI_20011.xml
    MVI_20012.xml
    MVI_20032.xml
    MVI_20013.xml
    ...
  val
    MVI_39031.xml
    MVI_39051.xml
    MVI_39211.xml
    MVI_39271.xml
    ...
```

### Script locations 
Script locations and tasks they perform are given below
* `main.py` - performs speed tracking
* `train.py` - trains the prepared data on yolov8 model
* `preprocessor/parser.py` - entry point for parsing, parses labels as needed by the yolo format; xml -> txt
* `preprocessor/image_preprocess.py` - preprocessor images, no need to call separately, invoked when running preprocessor/parser.py
* `setup.sh` - automates dataset installation, unzipping and parsing commands.

utils

* `utils/path.py` - utility file for tasks related to file handling
* `utils/draw.py` - useful drawing utils

everything from writing the preprocessing code, training the model, evaluating was done from the start.