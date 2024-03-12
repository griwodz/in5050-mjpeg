CFLAGS = -O3 -lm

TARGET = mjpeg_encoder

all: $(TARGET)

mjpeg_encoder: mjpeg_encoder.o
	nvcc $(CFLAGS) -o $@ $^

%.o: %.cu
	nvcc -g -c $^

clean:
	rm -f *.o mjpeg_encoder

foreman:
	./mjpeg_encoder -w 352 -h 288 -f 10 -o test.mjpeg ../foreman.yuv
