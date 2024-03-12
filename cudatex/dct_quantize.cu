#include <cuda.h>
#include <stdio.h>
#include <math.h>

#include "dct_quantize.h"
#include "mjpeg_encoder.h"

/* Global variables, for less redundancy processing */
uint32_t uv_width;
uint32_t uv_height;
uint32_t y_out_size;
uint32_t uv_out_size;

int16_t *Ydst;
int16_t *Udst;
int16_t *Vdst;

cudaArray *Y;
cudaArray *U;
cudaArray *V;
cudaTextureObject_t Y_tex;
cudaTextureObject_t U_tex;
cudaTextureObject_t V_tex;

__device__ __constant__ float cosv[8][8] = 
{{1.00000000000000000000f,0.98078525066375732422f,0.92387950420379638672f,0.83146959543228149414f,0.70710676908493041992f,0.55557024478912353516f,0.38268342614173889160f,0.19509032368659973145f},
 {1.00000000000000000000f,0.83146959543228149414f,0.38268342614173889160f,-0.19509032368659973145f,-0.70710676908493041992f,-0.98078525066375732422f,-0.92387950420379638672f,-0.55557024478912353516f},
 {1.00000000000000000000f,0.55557024478912353516f,-0.38268342614173889160f,-0.98078525066375732422f,-0.70710676908493041992f,0.19509032368659973145f,0.92387950420379638672f,0.83146959543228149414f},
 {1.00000000000000000000f,0.19509032368659973145f,-0.92387950420379638672f,-0.55557024478912353516f,0.70710676908493041992f,0.83146959543228149414f,-0.38268342614173889160f,-0.98078525066375732422f},
 {1.00000000000000000000f,-0.19509032368659973145f,-0.92387950420379638672f,0.55557024478912353516f,0.70710676908493041992f,-0.83146959543228149414f,-0.38268342614173889160f,0.98078525066375732422f},
 {1.00000000000000000000f,-0.55557024478912353516f,-0.38268342614173889160f,0.98078525066375732422f,-0.70710676908493041992f,-0.19509032368659973145f,0.92387950420379638672f,-0.83146959543228149414f},
 {1.00000000000000000000f,-0.83146959543228149414f,0.38268342614173889160f,0.19509032368659973145f,-0.70710676908493041992f,0.98078525066375732422f,-0.92387950420379638672f,0.55557024478912353516f},
 {1.00000000000000000000f,-0.98078525066375732422f,0.92387950420379638672f,-0.83146959543228149414f,0.70710676908493041992f,-0.55557024478912353516f,0.38268342614173889160f,-0.19509032368659973145f}
};

#define COSUV(i, j, k, l) ((float) (cosv[k][i] * cosv[l][j]))

/* We concatenate the 3 quantisation tables.
 * We can use offsets when accessing. This table will not
 * be changed, so we put it in constant memory
 */
__device__ __constant__ float quanttbl_gpu[192] =
{
  /*___ Y quantization table */
  16, 11, 12, 14, 12, 10, 16, 14,
  13, 14, 18, 17, 16, 19, 24, 40,
  26, 24, 22, 22, 24, 49, 35, 37,
  29, 40, 58, 51, 61, 30, 57, 51,
  56, 55, 64, 72, 92, 78, 64, 68,
  87, 69, 55, 56, 80, 109, 81, 87,
  95, 98, 103, 104, 103, 62, 77, 113,
  121, 112, 100, 120, 92, 101, 103, 99,
  /*___ U quantization table */
  17, 18, 18, 24, 21, 24, 47, 26,
  26, 47, 99, 66, 56, 66, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  /*___ V quantization table (same as U actually) */
  17, 18, 18, 24, 21, 24, 47, 26,
  26, 47, 99, 66, 56, 66, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99
};


/* The DCT is performed on the device. Use the DCT algorithm from precode.
 * ONE thread will work on ONE pixel.
 */
__global__ void dct_quantize( cudaTextureObject_t ImgSrc, int16_t *out_data, uint32_t padwidth, uint32_t width, uint32_t quant_offset )
{
    int i,j;
    float dct = 0;
    int yb = blockIdx.y * 8;
    int xb = blockIdx.x * 8;

    /* Get the appropriate quantization table, by offset into quanttbl_gpu. */
    float *quant = &quanttbl_gpu[quant_offset << 6];

    /* The temporary block should go in shared memory... Much faster than global! :) */
    __shared__ float tmp_block[8][8];

    /* Get pixel from texture memory and put it in shared memory */
    tmp_block[threadIdx.y][threadIdx.x] = tex2D<uint8_t>( ImgSrc, (xb+threadIdx.x), (yb+threadIdx.y) );
  
    /* Sync all threads, and kick off the 8x8 blocks */
    __syncthreads();

    for( i = 0; i < BLOCK_SIZE; i++ ) {
        for( j = 0; j < BLOCK_SIZE; j++ ) {
    	    dct += (tmp_block[i][j]-128.0f) * COSUV(threadIdx.x,threadIdx.y,i,j);
        }
    }

    float a1 = !(threadIdx.y) ? M_SQRT1_2 : 1.0f;
    float a2 = !(threadIdx.x) ? M_SQRT1_2 : 1.0f;

    dct *= a1*a2/4.0f;

    out_data[(yb+threadIdx.x)*padwidth+(xb+threadIdx.y)] = (int16_t)(floor(0.5f + dct / (quant[threadIdx.x*BLOCK_SIZE+threadIdx.y])));
}

/* Init function is called from main in mjpeg_encoder.c. When doing this, we can reduce
 * some overhead, and not have to start ut and initialize the GPU everytime we call dct.
 *
 * Inti sets up grids and thread grids for the YUV format.
 * We also set up necessary variables for processing the frames
 * Init is done by the host!
 */
__host__ void gpu_init()
{
  uv_width	    = (width / 2);
  uv_height	    = (height / 2);

  /* Multiply by 2, since ipupt is uint8_t and output is int16_t */
  uv_out_size	    = (uv_comp_size * 2);
  y_out_size	    = (y_comp_size  * 2);


  /* Handle u and v components, not the same size as Y */
  int uv_width_pad  = (ypw / 2);
  int uv_height_pad = (yph / 2);
  int uv_areal_pad  = (upw * uph);
  int uv_areal_pad2 = (uv_areal_pad * 2);

  /* Do the memory! */

  /* The texture memory for the source image.               *
   * Note that we use C++ here to make a life a bit easier. */
  cudaChannelFormatDesc chartex = cudaCreateChannelDesc<uint8_t>();

  /* Allocate memory on the device for the input data (YUV components) */
  cudaMallocArray(&Y, &chartex, width, height);
  cudaMallocArray(&U, &chartex, uv_width_pad, uv_height_pad);
  cudaMallocArray(&V, &chartex, uv_width_pad, uv_height_pad);

  cudaTextureDesc  tex_desc;
  memset( &tex_desc, 0, sizeof(cudaTextureDesc) );
  tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
  tex_desc.addressMode[0]   = cudaAddressModeClamp;
  tex_desc.addressMode[1]   = cudaAddressModeClamp;
  tex_desc.addressMode[2]   = cudaAddressModeClamp;
  tex_desc.readMode         = cudaReadModeElementType; // read original type (1-byte)
  tex_desc.filterMode       = cudaFilterModePoint;     // no interpolation

  cudaResourceDesc res_desc;
  res_desc.resType         = cudaResourceTypeArray;
  res_desc.res.array.array = Y;
  cudaCreateTextureObject( &Y_tex, &res_desc, &tex_desc, 0 );
  res_desc.res.array.array = U;
  cudaCreateTextureObject( &U_tex, &res_desc, &tex_desc, 0 );
  res_desc.res.array.array = V;
  cudaCreateTextureObject( &V_tex, &res_desc, &tex_desc, 0 );

  /* If we must pad, set all values to 128 */
  cudaMemset(U,128,uv_areal_pad);
  cudaMemset(V,128,uv_areal_pad);

  /* Allocate memory on the device for the output */
  cudaError_t err = cudaMalloc((void **) &Ydst, y_out_size);
  if( err != cudaSuccess ) {
    printf( "%s\n", cudaGetErrorString(err) );
  }
  cudaMalloc((void **) &Udst, uv_areal_pad2);
  cudaMalloc((void **) &Vdst, uv_areal_pad2);
}

/* Host code. This is the function called from mjpeg_encoder.c.
 * Handles all copying, the binding of texture memory and calls
 * the dct_quantize function. It also handles the output data.
 */
__host__ void gpu_dct_quantize(yuv_t *image, dct_t *out)
{
  /* Copy input to the device */
  cudaMemcpy2DToArray( Y, 0, 0,
                       image->Y, width * sizeof(uint8_t),
                       width * sizeof(uint8_t), height,
                       cudaMemcpyHostToDevice );
  cudaMemcpy2DToArray( U, 0, 0,
                       image->U, uv_width * sizeof(uint8_t),
                       uv_width * sizeof(uint8_t), uv_height,
                       cudaMemcpyHostToDevice );
  cudaMemcpy2DToArray( V, 0, 0,
                       image->V, uv_width * sizeof(uint8_t),
                       uv_width * sizeof(uint8_t), uv_height,
                       cudaMemcpyHostToDevice );

  /* Init blocks and threads for GPU,
   * One grid for Y, and one for U and V
   */
  dim3 block_grid_Y;
  dim3 block_grid_UV;
  dim3 thread_grid;

  /* Block grid: NUM_8x8BLOCKSxNUM_8x8BLOCKS Y component */
  block_grid_Y.y    = height >> BLOCK_SIZE_LOG;
  block_grid_Y.x    = width >> BLOCK_SIZE_LOG;

  /* Block grid: NUM_8x8BLOCKSxNUM_8x8BLOCKS U and V component */
  block_grid_UV.y   = uph >> BLOCK_SIZE_LOG;
  block_grid_UV.x   = upw >> BLOCK_SIZE_LOG;

  /* Grid size: 8x8 pixels */
  thread_grid.x	    = BLOCK_SIZE;
  thread_grid.y	    = BLOCK_SIZE;

  /* Select texture object for Y and call dct_quantize with Y grid size */
  dct_quantize
    <<<block_grid_Y,  thread_grid>>>
    ( Y_tex, Ydst, ypw, width, Y_QUANT  );

  /* Select texture object for U and call dct_quantize with UV (U) grid size */
  dct_quantize
    <<<block_grid_UV, thread_grid>>>
    ( U_tex, Udst, upw, uv_width, U_QUANT );

  /* Select texture object for V and call dct_quantize with UV (V) grid size */
  dct_quantize
    <<<block_grid_UV, thread_grid>>>
    ( V_tex, Vdst, vpw, uv_width, V_QUANT );

  /* Copy back to the host from the device memory  */
  cudaMemcpy(out->Ydct, Ydst, y_out_size,  cudaMemcpyDeviceToHost);
  cudaMemcpy(out->Udct, Udst, uv_out_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(out->Vdct, Vdst, uv_out_size, cudaMemcpyDeviceToHost);
}

/* Clean up! ;) */
__host__ void gpu_cleanup()
{
  cudaDestroyTextureObject( Y_tex );
  cudaDestroyTextureObject( U_tex );
  cudaDestroyTextureObject( V_tex );

  cudaFreeArray(Y);
  cudaFreeArray(U);
  cudaFreeArray(V);

  cudaFree(Ydst);
  cudaFree(Udst);
  cudaFree(Vdst);
}
