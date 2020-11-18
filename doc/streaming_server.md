# Streaming server
The streaming server is used to observe detection images which are pushed from the detection node. 

Run the container in debug mode:
```shell
docker run --rm -it --network=host aler9/rtsp-simple-server
``` 
Run with authentication in background:
```shell
docker run -d --network=host \
--env RTSP_PATHS_ALL_PUBLISHUSER=pusher \
--env RTSP_PATHS_ALL_PUBLISHPASS=myrandompass \
--env RTSP_PATHS_ALL_READUSER=viewer \
--env RTSP_PATHS_ALL_READPASS=myrandompass \
aler9/rtsp-simple-server
``` 

Push stream from webcam of a client:
```shell
ffmpeg -r 5 \
-s 640x480 \
-f v4l2 \
-input_format mjpeg \
-i /dev/video0 \
-c:v copy \
-f rtsp \
rtsp://pusher:myrandompass@stream_server_ip_or_fqnd:8554/mystream
```

Read stream from a third device:
```shell
ffplay -i rtsp://viewer:myrandompass@stream_server_ip_or_fqnd:8554/mystream
```