#!/bin/bash

# Install and configure virtual camera device
#sudo apt install v4l2loopback-dkms
#sudo modprobe v4l2loopback

file_to_stream=$1
device=$2

temp_file="/tmp/${device: -1}_$file_to_stream"
echo $temp_file
rsync -a $file_to_stream $temp_file
ffmpeg -re -stream_loop -1 -i $temp_file -pix_fmt yuv420p -f v4l2 $device

# -re  - use video's native framerate
# -stream_loop -1    - infinitely loop the video
# ffmpeg -re -stream_loop -1 -i scene_1_merged.mp4 -pix_fmt yuv420p -f v4l2 /dev/video2
