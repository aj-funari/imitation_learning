# imitation_learning

Script #1: Data collection
  1) Control Jackal
  * PS4 Controller
  * Command line
      * rostopic pub /cmd_vel geometry_msgs/Twist -r 10 '[1.0, 0.0, 0.0]' '[0.0, 0.0, 1.0]'
  2) Save images to hard drive
  * Subscribe to cmd_vel
  * Subscribe to image

Script #2: Create customized dataset
* https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

Script #3: Create model (skeleton)

Script #4: Train model
* customized dataset
* model (skeleton)
* evaluate results 
