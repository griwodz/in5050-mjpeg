#include <cuda.h>
#include <stdio.h>
#include <math.h>

#include "dct_quantize.h"
#include "mjpeg_encoder.h"
#include "cosv.h"
#include "quant_table.h"

/* Global variables, for less redundancy processing */
uint32_t uv_width;
uint32_t uv_height;
uint32_t y_out_size;
uint32_t uv_out_size;

/* The DCT is performed on the device. Use the DCT algorithm from precode.
 * ONE thread will work on ONE pixel.
 */
__global__ void dct_quantize( uint8_t *inn_data, int16_t *out_data, uint32_t padwidth, uint16_t width, uint32_t quant_offset )
{
    int i,j;
    float dct = 0;
    int yb = blockIdx.y * BLOCK_SIZE;
    int xb = blockIdx.x * BLOCK_SIZE;

    /* Get the appropriate quantization table, by offset into quanttbl_gpu. */
    float *quant = &quanttbl_gpu[quant_offset << 6];

    /* The temporary block should go in shared memory... Much faster than global! :) */
    __shared__ float tmp_block[BLOCK_SIZE][BLOCK_SIZE];

    /* Get pixel from global memory and put it in shared memory */
    tmp_block[threadIdx.y][threadIdx.x] = inn_data[(yb+threadIdx.y)*width+(xb+threadIdx.x)];

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
}


/* Host code. This is the function called from mjpeg_encoder.c.
 * Handles all copying, the binding of texture memory and calls
 * the dct_quantize function. It also handles the output data.
 */
__host__ void gpu_dct_quantize(yuv_t *image, dct_t *out)
{
  /* Copy input to the device */
  uint8_t *Yinn = image->Y;
  uint8_t *Uinn = image->U;
  uint8_t *Vinn = image->V;
  int16_t *Ydst = out->Ydct;
  int16_t *Udst = out->Udct;
  int16_t *Vdst = out->Vdct;

  cudaStreamAttachMemAsync(NULL, Yinn, 0, cudaMemAttachGlobal);
  cudaStreamAttachMemAsync(NULL, Uinn, 0, cudaMemAttachGlobal);
  cudaStreamAttachMemAsync(NULL, Vinn, 0, cudaMemAttachGlobal);
  cudaStreamAttachMemAsync(NULL, Ydst, 0, cudaMemAttachGlobal);
  cudaStreamAttachMemAsync(NULL, Udst, 0, cudaMemAttachGlobal);
  cudaStreamAttachMemAsync(NULL, Vdst, 0, cudaMemAttachGlobal);

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

  /* Launch dct_quantize kernel with Y grid size */
  dct_quantize <<<block_grid_Y, thread_grid>>> ( Yinn, Ydst, ypw, width, Y_QUANT );
  cudaStreamAttachMemAsync(NULL, Ydst, 0, cudaMemAttachHost);

  /* Launch dct_quantize kernel with UV (U) grid size */
  dct_quantize <<<block_grid_UV, thread_grid>>> ( Uinn ,Udst, upw, uv_width, U_QUANT );
  cudaStreamAttachMemAsync(NULL, Udst, 0, cudaMemAttachHost);

  /* Launch dct_quantize kernel with UV (V) grid size */
  dct_quantize <<<block_grid_UV, thread_grid>>> ( Vinn, Vdst, vpw, uv_width, V_QUANT );
  cudaStreamAttachMemAsync(NULL, Vdst, 0, cudaMemAttachHost);

  cudaDeviceSynchronize();

  cudaStreamAttachMemAsync(NULL, Yinn, 0, cudaMemAttachHost);
  cudaStreamAttachMemAsync(NULL, Uinn, 0, cudaMemAttachHost);
  cudaStreamAttachMemAsync(NULL, Vinn, 0, cudaMemAttachHost);
}

/* Clean up! ;) */
__host__ void gpu_cleanup()
{
}
