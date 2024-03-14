ARCH=-gencode arch=compute_72,code=sm_72 \
     -gencode arch=compute_72,code=compute_72
CFLAGS = --std=c++17 -O0 -g $(ARCH) \
         -Wno-deprecated-gpu-targets \
	 -Wreorder \
	 -Xcompiler "-march=armv8-a" -Xptxas --warn-on-spills

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

