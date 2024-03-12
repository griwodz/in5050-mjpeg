---- IN5050 - Motion JPEG Encoder ----

COMPILE:

make


EXAMPLE USE:

./mjpeg_encoder -w 352 -h 288 -f 10 -o test.mjpeg foreman.yuv

w = Width of input video
h = Height of input video
o = Name on output file

f = Optional argument for limiting the number of frames.

Output MJPEG videos can be viewed in "mplayer" or "vlc"


INPUT VIDEOS:

Can be found on the lab-machines:
/mnt/sdcard

Can also be downloaded from here (need to convert y4m to yuv):
https://media.xiph.org/video/derf/

Encoder accepts YUV 4:2:0 frames.
