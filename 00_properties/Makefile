CFLAGS = -O3 -g

TARGET = prop

all: $(TARGET)

prop: prop.o
	nvcc $(CFLAGS) -o $@ $^

%.o: %.cu
	nvcc $(CFLAGS) -c $^

test: prop
	./prop

clean:
	rm -f *.o
	rm -f prop

