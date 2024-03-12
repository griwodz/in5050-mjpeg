CFLAGS = -O3 -lm

TARGET = mjpeg_encoder

all: $(TARGET)

mjpeg_encoder: mjpeg_encoder.o
	nvcc $(CFLAGS) -o $@ $^

%.o: %.cu
	nvcc -g -c $^

clean:
	rm -f *.o
	rm -f mjpeg_encoder
	rm -f test.mjpeg

foreman: mjpeg_encoder
	./mjpeg_encoder -w 352 -h 288 -f 20 -o test.mjpeg ../foreman.yuv

play:
	# -vo caca means colored ASCII art, -vo aa means black and white ASCII art
	mplayer -vo aa test.mjpeg

