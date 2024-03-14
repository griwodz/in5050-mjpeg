CFLAGS = --std=c++17 -O3 -g

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

test.mjpeg: mjpeg_encoder
	./mjpeg_encoder -w 352 -h 288 -f 20 -o test.mjpeg ../foreman.yuv

foreman: test.mjpeg

play: test.mjpeg
	# -vo caca means colored ASCII art, -vo aa means black and white ASCII art
	mplayer -vo aa test.mjpeg

