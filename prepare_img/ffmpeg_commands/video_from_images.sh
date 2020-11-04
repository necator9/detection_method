#!/bin/bash

in_dir="/mnt/data_partition/experiments/sources/clf_test/night/added_to_dataset/sc_2_parking_pg_02/src_320x240_grayscale/"
out_file="/mnt/data_partition/experiments/sources/clf_test/night/added_to_dataset/sc_2_parking_pg_02/src_320x240_grayscale.mp4"

fps_scene_1=5
fps_scene_2=10
#ffmpeg -r 5 -f image2 -s 424x240 -i $in_dir/%04d.jpeg -vf hue=s=0 -vcodec libx264 -crf 25 -pix_fmt yuv420p -y $out_file
ffmpeg -r $fps_scene_2 -f image2 -s 424x240 -i $in_dir/%04d.jpeg -vf hue=s=0 -vcodec libx264 -crf 0 \
-preset veryslow -y $out_file


#    -r is the framerate (fps)
#    -crf is the quality, lower means better quality, 15-25 is usually good
#    -s is the resolution
#    -pix_fmt yuv420p specifies the pixel format, change this as needed
# Crop aspect ratio
# ffmpeg -i sc_1_parking_c_01_16:9_424x240.mp4 -vf crop=iw-104 -vcodec libx264 -crf 0 -preset veryslow -y destination.mp4
# Encode into a raw_video
# ffmpeg -f concat -safe 0 -i mergelist.txt -r 4 -s 320x240 -vf hue=s=0 -vcodec rawvideo -pix_fmt gray -y car_night_merged_reen1.mp4