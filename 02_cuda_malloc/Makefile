CC = nvcc
ARCH=-gencode arch=compute_72,code=sm_72 \
     -gencode arch=compute_72,code=compute_72
CFLAGS = --std=c++17 -O0 -g
CUFLAGS = $(CFLAGS) -G $(ARCH) -rdc=true

all: mjpeg_gpu

mjpeg_gpu: mjpeg_encoder.o dct_quantize.o tables.o cosv.o quant_table.o diskwrite.o
	$(CC) $(CUFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $<

%.o: %.cu
	$(CC) $(CUFLAGS) -c $<

clean:
	rm -f *.o mjpeg_gpu
	rm -rf mjpeg_gpu.dSYM
	rm -rf test.mjpeg
	rm -f Report.ncu-rep

foreman: test.mjpeg

test.mjpeg: mjpeg_gpu
	# ./mjpeg_gpu -w 352 -h 288 -f 10 -o test.mjpeg foreman.yuv
	./mjpeg_gpu -w 352 -h 288 -o test.mjpeg ../foreman.yuv

profile:
	# sudo /usr/local/cuda-12.2/bin/ncu --target-processes all -f -o Report ./mjpeg_gpu -w 352 -h 288 -f 20 -o test.mjpeg ../foreman.yuv
	sudo /usr/local/cuda-12.2/bin/ncu --profile 0 -f -o Report ./mjpeg_gpu -w 352 -h 288 -f 20 -o test.mjpeg ../foreman.yuv

play: test.mjpeg
	# -vo caca means colored ASCII art, -vo aa means black and white ASCII art
	mplayer -vo aa test.mjpeg
