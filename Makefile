clean:
	make -C 00_properties clean
	make -C 01_cpu clean
	make -C 02_cuda_malloc clean
	make -C 03_cuda_managed clean
	make -C cuda-texture clean

