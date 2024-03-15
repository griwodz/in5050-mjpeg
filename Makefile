clean:
	make -C 00_properties clean
	make -C 01_cpu clean
	make -C 02_cuda_malloc clean
	make -C 03_cuda_managed clean
	make -C 04_cuda_texture clean
	make -C 05_cuda_mgd_cg clean
	make -C 06_cuda_mgd_stream clean
	make -C 07_cuda_mgd_sys clean
	make -C 08_cuda_mgd_zcpy clean
	rm -f report-*

