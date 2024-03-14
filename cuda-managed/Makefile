CC = nvcc
ARCH=
CFLAGS = -O0 -g $(ARCH)
CUFLAGS = -G $(ARCH)  -rdc=true
# CUFLAGS = -G $(ARCH)
LDFLAGS= $(ARCH) # -lnvToolsExt

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
	# ./mjpeg_gpu -w 352 -h 288 -f 10 -o test.mjpeg foreman.yuv
	./mjpeg_gpu -w 352 -h 288 -o test.mjpeg foreman.yuv

