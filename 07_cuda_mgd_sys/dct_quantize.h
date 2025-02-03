#ifndef DCT_QUANTIZE_H
#define DCT_QUANTIZE_H

#include <stdint.h>
#include "mjpeg_encoder.h"

enum
{
  Y_QUANT,
  U_QUANT,
  V_QUANT
};

/* CUDA-stuff... */
#ifdef __cplusplus
extern "C"
{
#endif
  void gpu_dct_quantize(yuv_t *image, dct_t *out);
  void gpu_init();
  void gpu_cleanup();
#ifdef __cplusplus
}
#endif

#endif /* dct_quantize.h */
