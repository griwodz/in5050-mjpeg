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
static bool read_yuv(FILE *file, yuv_t *image)
{
    size_t len = 0;

    /* Read Y' */
    len += fread(image->Y, 1, width*height, file);
    if (ferror(file))
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    /* Read U */
    len += fread(image->U, 1, (width*height)/4, file);
    if (ferror(file))
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    /* Read V */
    len += fread(image->V, 1, (width*height)/4, file);
    if (ferror(file))
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    if (len != width*height*1.5)
    {
        printf("Reached end of file.\n");
        return false;
    }

    return true;
}

static void encode(yuv_t *image, dct_t *out)
{
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
    uv_comp_size	= (width * height) / 4;

    input_file = argv[optind];

    if (limit_numframes)
    {
      printf("Limited to %d frames.\n", limit_numframes);
    }

    FILE *infile = fopen(input_file, "rb");

    if (infile == NULL)
    {
      perror("fopen");
      exit(EXIT_FAILURE);
    }

    /* Allocate data before read and encode loop */
    yuv_t *image = (yuv_t*)malloc(sizeof(yuv_t));
    cudaMallocManaged((void **) &image->Y, width*height);
    cudaMallocManaged((void **) &image->U, width*height);
    cudaMallocManaged((void **) &image->V, width*height);

    dct_t *out = (dct_t*)malloc(sizeof(dct_t));
    cudaMallocManaged((void **) &out->Ydct, yph*ypw*(sizeof(int16_t)) );
    cudaMallocManaged((void **) &out->Udct, uph*upw*(sizeof(int16_t)) );
    cudaMallocManaged((void **) &out->Vdct, vph*vpw*(sizeof(int16_t)) );

    int numframes = 0;

    /* INIT GPU - Gentlemen, please start you multiprocessors */
    gpu_init();

    /* Parse input files */
    while (!feof(infile))
    {
        bool ok = read_yuv(infile, image);

        if (!ok)
        {
            break;
        }

        printf("Encoding frame %d, ", numframes);
        encode(image, out);
        printf("Done!\n");

        ++numframes;

        if (limit_numframes && numframes >= limit_numframes)
        {
            break;
        }
    }

    /* Clean up all memory used on the GPU */
    gpu_cleanup();
    
    fclose(outfile);
    fclose(infile);

    cudaFree(image->Y);
    cudaFree(image->U);
    cudaFree(image->V);
    free( image );

    cudaFree(out->Ydct);
    cudaFree(out->Udct);
    cudaFree(out->Vdct);
    free( out );

    exit (EXIT_SUCCESS);
}
