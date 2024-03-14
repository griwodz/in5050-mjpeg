#ifndef MJPEG_ENCODER_H
#define MJPEG_ENCODER_H

#include <stdint.h>

#define MAX_FILELENGTH 200
#define DEFAULT_OUTPUT_FILE "a.mjpeg"

#define ISQRT2 0.70710678118654f
#define PI 3.14159265358979
#define ILOG2 1.442695040888963 // 1/log(2);

#define COLOR_COMPONENTS 3

#define YX 2
#define YY 2
#define UX 1
#define UY 1
#define VX 1
#define VY 1

#define MIN(a,b) ((a) < (b) ? (a) : (b))

extern FILE*    outfile;
extern uint32_t height;
extern uint32_t width;
extern uint32_t yph;
extern uint32_t ypw;
extern uint32_t uph;
extern uint32_t upw;
extern uint32_t vph;
extern uint32_t vpw;

struct yuv
{
  uint8_t *Y;
  uint8_t *U;
  uint8_t *V;
};

struct dct
{
  int16_t *Ydct;
  int16_t *Udct;
  int16_t *Vdct;
  int pic_id;
};

typedef struct yuv yuv_t;
typedef struct dct dct_t;


/* Some global variables...
 * These are shared between the host and the GPU
 */

extern uint32_t y_comp_size;
extern uint32_t uv_comp_size;

#endif /* mjpeg_encoder.h */
