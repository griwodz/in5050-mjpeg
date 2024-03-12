CC = nvcc
CFLAGS = -O0 -g \
          -gencode arch=compute_30,code=sm_30 \
	  -gencode arch=compute_30,code=compute_30
CUFLAGS = -G -rdc=true \
          -gencode arch=compute_30,code=sm_30 \
	  -gencode arch=compute_30,code=compute_30
	# RDC allow "separable compilation" - a CUDA kernel can use code and data from several .cu files
LDFLAGS= -gencode arch=compute_30,code=sm_30 \
          -gencode arch=compute_30,code=compute_30
	  # -lnvToolsExt

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
	./mjpeg_gpu -w 352 -h 288 -f 10 -o test.mjpeg /mnt/sdcard/foreman.yuv

