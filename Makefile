clean:
	make -C 01_cpu clean
	make -C cuda-malloc clean
	make -C cuda-managed clean
	make -C cuda-texture clean
	make -C cudatex clean
	make -C managed clean
