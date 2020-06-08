#!/bin/bash

# Install and configure virtual camera device
#sudo apt install v4l2loopback-dkms
#sudo modprobe v4l2loopback

file_to_stream=$HOME'/scene_1_merged.mp4'
device='/dev/video1'

ffmpeg -re -stream_loop -1 -i $file_to_stream -pix_fmt yuv420p -f v4l2 $device

# -re  - use video's native framerate
# -stream_loop -1    - infinitely loop the video
# ffmpeg -re -stream_loop -1 -i scene_1_merged.mp4 -pix_fmt yuv420p -f v4l2 /dev/video2