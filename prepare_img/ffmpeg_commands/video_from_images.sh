#!/bin/bash

in_dir="/home/ivan/Documents/datasets/real/in/sc_2_parking_c_01/src_320x240_grayscale/"
out_file="/home/ivan/Documents/datasets/real/in/sc_2_parking_c_01/src_320x240_grayscale_4x3.mp4"

fps_scene_1=5   # TZK1
fps_scene_2=10  # TZK2
#ffmpeg -r 5 -f image2 -s 424x240 -i $in_dir/%04d.jpeg -vf hue=s=0 -vcodec libx264 -crf 25 -pix_fmt yuv420p -y $out_file
ffmpeg -r $fps_scene_2 -f image2 -s 424x240 -i $in_dir/%04d.jpeg -vf hue=s=0 -vcodec libx264 -crf 0 \
-preset veryslow -pix_fmt yuv420p -y $out_file


#    -r is the framerate (fps)
#    -crf is the quality, lower means better quality, 15-25 is usually good
#    -s is the resolution
#    -pix_fmt yuv420p specifies the pixel format, change this as needed
# Crop aspect ratio
# ffmpeg -i sc_1_parking_c_01_16:9_424x240.mp4 -vf crop=iw-104 -vcodec libx264 -crf 0 -preset veryslow -y destination.mp4
# Encode into a raw_video
# ffmpeg -f concat -safe 0 -i mergelist.txt -r 4 -s 320x240 -vf hue=s=0 -vcodec rawvideo -pix_fmt gray -y car_night_merged_reen1.mp4

# Change FPS, crop aspect ratio and change resolution
# ffmpeg -i corrected_movement_0_11_8.73.mp4 -vf crop=iw-104,fps=4,hue=s=0 -vcodec libx264 -crf 0 -preset veryslow -s 320x240 -y destination.mp4