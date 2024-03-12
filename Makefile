uname_p := $(shell uname -p)
CC = gcc

ifeq ($(uname_p),x86_64)
CFLAGS = -Wall -O3 -lm
else
CFLAGS = -Wall -march=armv8.2-a -O3 -lm
endif

LIBS = -lm
TARGET = mjpeg_encoder

all: $(TARGET)

$(TARGET): $(TARGET).o
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).c $(LIBS)

clean:
	rm -f *.o mjpeg_encoder

foreman:
	./mjpeg_encoder -w 352 -h 288 -f 10 -o test.mjpeg foreman.yuv
