ARCHFLAGS=-gencode arch=compute_72,code=sm_72 \
	  -gencode arch=compute_72,code=compute_72
_RCHFLAGS=-gencode arch=compute_87,code=sm_87 \
	  -gencode arch=compute_87,code=compute_87

CFLAGS = -O0 -g -G $(ARCHFLAGS)
LDFLAGS= $(ARCHFLAGS)

# CFLAGS = -O3 -g

TARGET = mjpeg_encoder

all: $(TARGET)

mjpeg_encoder: mjpeg_encoder.o diskwrite.o
	nvcc $(CFLAGS) -o $@ $^

%.o: %.cu
	nvcc $(CFLAGS) -c $^

clean:
	rm -f *.o
	rm -f mjpeg_encoder
	rm -f test.mjpeg

foreman: mjpeg_encoder
	./mjpeg_encoder -w 352 -h 288 -f 20 -o test.mjpeg ../foreman.yuv

play:
	# -vo caca means colored ASCII art, -vo aa means black and white ASCII art
	mplayer -vo aa test.mjpeg

