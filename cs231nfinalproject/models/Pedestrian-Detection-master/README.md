# Pedestrain Detection
A Python Program that uses OpenCV's haarcascade to Detect Pedestrians

## Requirements
Language Used = Python3<br />
Modules/Packages used:
* cv2
* datetime
* optparse
* time
* colorama
<!-- -->
Install the dependencies:
```bash
pip install -r requirements.txt
```

## Input
It takes the following arguments from the command that is used to run the Python Program:
* '-i', "--image" : Path to the Image file (If not specified, will take frame from Camera Stream)
* '-p', "--cascade-file-pedestrians" : Path to cascade File for Pedestrian Detection
* '-s', "--scale-factor" : Scale Factor
* '-m', "--min-neighbors" : Minimum Neighbors

## Output
Image/Live Video Stream from the Camera with Detected Pedestrains.<br />
On the Command Line Interface, it displays the number of Pedestrains Detected.

### Note
If the output was not what you expected, try to adjust the **scale-factor** and **min-neighbors** value.