#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

#include "mjpeg_encoder.h"
#include "dct_quantize.h"
#include "tables.h"
#include "diskwrite.h"

static char *output_file;
static char *input_file;
// static float cosuv[8][8][8][8];

FILE *outfile;

static int limit_numframes = 0;

// static uint32_t bit_buffer = 0;
// static uint32_t bit_buffer_width = 0;

uint32_t height;
uint32_t width;

uint32_t y_comp_size;
uint32_t uv_comp_size;

uint32_t yph;
uint32_t ypw;
uint32_t uph;
uint32_t upw;
uint32_t vph;
uint32_t vpw;

/* getopt */
extern int optind;
extern char *optarg;

/* Read YUV frames */
static yuv_t* read_yuv(FILE *file)
{
    size_t len = 0;
    yuv_t *image = (yuv_t*)malloc(sizeof(yuv_t));


    /* Read Y' */
    image->Y = (uint8_t*)malloc(width*height);
    len += fread(image->Y, 1, width*height, file);
    if (ferror(file))
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    /* Read U */
    image->U = (uint8_t*)malloc(width*height);
    len += fread(image->U, 1, (width*height)/4, file);
    if (ferror(file))
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    /* Read V */
    image->V = (uint8_t*)malloc(width*height);
    len += fread(image->V, 1, (width*height)/4, file);
    if (ferror(file))
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    if (len != width*height*1.5)
    {
        printf("Reached end of file.\n");
        return NULL;
    }

    return image;
}

static void encode(yuv_t *image)
{
  dct_t *out = (dct_t*)malloc(sizeof(dct_t));
  out->Ydct = (int16_t*)malloc(yph*ypw*(sizeof(*out->Ydct)));
  out->Udct = (int16_t*)malloc(uph*upw*(sizeof(*out->Udct)));
  out->Vdct = (int16_t*)malloc(vph*vpw*(sizeof(*out->Vdct)));

  memset(out->Ydct, 0, sizeof(int16_t) * yph * ypw);
  memset(out->Udct, 0, sizeof(int16_t) * uph * upw);
  memset(out->Vdct, 0, sizeof(int16_t) * vph * vpw);

  /* We have this nice GPU up and running! Let's do some DCT! */
  gpu_dct_quantize(image, out);

  /* Write headers */
  /* Start Of Image */
  write_SOI();
  /* Define Quantization Table(s) */
  write_DQT();
  /* Start Of Frame 0(Baseline DCT) */
  write_SOF0();
  /* Define Huffman Tables(s) */
  write_DHT();
  /* Start of Scan */
  write_SOS();

  write_interleaved_data(out);

  /* End Of Image */
  write_EOI();

  free(out->Ydct);
  free(out->Udct);
  free(out->Vdct);
  free(out);
}

static void print_help()
{
    fprintf(stderr, "Usage: ./mjpeg_encoder [options] input_file\n");
    fprintf(stderr, "Commandline options:\n");
    fprintf(stderr, "  -h                             height of images to compress\n");
    fprintf(stderr, "  -w                             width of images to compress\n");
    fprintf(stderr, "  -o                             Output file (.mjpg)\n");
    fprintf(stderr, "  [-f]                           Limit number of frames to encode\n");
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
  int c;
  yuv_t *image;

  if (argc == 1)
  {
    print_help();
    exit(EXIT_FAILURE);
  }

    while((c = getopt(argc, argv, "h:w:o:f:i:")) != -1)
    {
        switch(c)
        {
        case 'h':
            height = atoi(optarg);
            break;
        case 'w':
            width = atoi(optarg);
            break;
        case 'o':
            output_file = optarg;
            break;
        case 'f':
            limit_numframes = atoi(optarg);
            break;
        default:
            print_help();
            break;
        }
    }

  if (optind >= argc)
  {
	  fprintf(stderr, "Error getting program options, try --help.\n");
	  exit(EXIT_FAILURE);
  }

  outfile = fopen(output_file, "wb");
  if (outfile == NULL)
  {
	  perror("fopen");
	  exit(EXIT_FAILURE);
  }

  /* Calculate the padded width and height */
  ypw = (uint32_t)(ceil(width/8.0f)*8);
  yph = (uint32_t)(ceil(height/8.0f)*8);
  upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
  uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
  vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
  vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

  /* Divide u and v by 4 */
  y_comp_size	  = width * height;
  uv_comp_size	  = y_comp_size >> 2;

  input_file = argv[optind];

  if (limit_numframes) {
	  printf("Limited to %d frames.\n", limit_numframes);
  }

  FILE *infile = fopen(input_file, "rb");

  if (infile == NULL)
  {
	  perror("fopen");
	  exit(EXIT_FAILURE);
  }

  int numframes = 0;


  /* INIT GPU - Gentlemen, please start you multiprocessors */
  gpu_init();

  /* Parse input files */
  while (!feof(infile))
  {
	image = read_yuv(infile);

	if (!image) {
		break;
	}

	printf("Encoding frame %d, ", numframes);
    encode(image);

    free(image->Y);
    free(image->U);
    free(image->V);
    free(image);

	printf("Done!\n");

	++numframes;
	if (limit_numframes && numframes >= limit_numframes) {
		break;
	}
  }

  /* Clean up all memory used on the GPU */
  gpu_cleanup();

  fclose(outfile);
  fclose(infile);

  exit (EXIT_SUCCESS);

}
