CC = nvcc

#
# CC 8.7: Jetson Orin Nano, Jetson Orin NX, Jetson AGX Orin
#
ARCHFLAGS=-gencode arch=compute_87,code=sm_87 \
	  -gencode arch=compute_87,code=compute_87

CFLAGS = -O0 -g $(ARCHFLAGS)
CUFLAGS= -G -rdc=true $(ARCHFLAGS) # RDC allow "separable compilation"
LDFLAGS= $(ARCHFLAGS)

all: mjpeg_gpu

%.o: %.cu
	$(CC) $(CFLAGS) $(CUFLAGS) -c $<

mjpeg_gpu: mjpeg_encoder.o dct_quantize.o tables.o cosv.o quant_table.o
	$(CC) $(CFLAGS) -o mjpeg_gpu $^ $(LDFLAGS)

clean:
	rm -f *.o mjpeg_gpu
	rm -rf mjpeg_gpu.dSYM
	rm -rf test.mjpeg

foreman:
	./mjpeg_gpu -w 352 -h 288 -f 100 -o test.mjpeg ../foreman.yuv

play:
	# -vo caca means colored ASCII art, -vo aa means black and white ASCII art
	mplayer -vo caca test.mjpeg
