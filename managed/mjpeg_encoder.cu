#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

#include "diskwrite.h"
#include "mjpeg_encoder.h"

static char *output_file;
static char *input_file;
FILE *outfile;

static int limit_numframes = 0;

uint32_t height;
uint32_t width;
uint32_t yph;
uint32_t ypw;
uint32_t uph;
uint32_t upw;
uint32_t vph;
uint32_t vpw;

/* getopt */
extern int optind;
extern char *optarg;

void fatality_test( cudaError_t err, const char* info )
{
    if( err == cudaSuccess ) return;

    fprintf( stderr, "%s : %s\n", info, cudaGetErrorString( err ) );

    exit( -1 );
}

/* Read YUV frames */
static yuv_t* read_yuv(FILE *file, yuv_t* image)
{
    size_t len = 0;

    /* Read Y' */
    len += fread(image->Y, 1, width*height, file);
    if( ferror(file) )
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    /* Read U */
    len += fread(image->U, 1, (width*height)/4, file);
    if( ferror(file) )
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    /* Read V */
    len += fread(image->V, 1, (width*height)/4, file);
    if( ferror(file) )
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    if( len != width*height*1.5 )
    {
        printf("Reached end of file.\n");
        return NULL;
    }

    return image;
}

static void dct_quantize(uint8_t *in_data, uint32_t width, uint32_t height,
        int16_t *out_data, uint32_t padwidth,
        uint32_t padheight, uint8_t *quantization)
{
    int y,x,u,v,j,i;

    /* Perform the DCT and quantization */
    for(y = 0; y < height; y += 8)
    {
        int jj = height - y;
        jj = MIN(jj, 8); // For the border-pixels, we might have a part of an 8x8 block

        for(x = 0; x < width; x += 8)
        {
            int ii = width - x;
            ii = MIN(ii, 8); // For the border-pixels, we might have a part of an 8x8 block

            //Loop through all elements of the block
            for(u = 0; u < 8; ++u)
            {
                for(v = 0; v < 8; ++v)
                {
                    /* Compute the DCT */
                    float dct = 0;
                    for(j = 0; j < jj; ++j)
                        for(i = 0; i < ii; ++i)
                        {
                            float coeff = in_data[(y+j)*width+(x+i)] - 128.0f;
                            dct += coeff * (float) (cos((2*i+1)*u*PI/16.0f) * cos((2*j+1)*v*PI/16.0f));
                        }

                    float a1 = !u ? ISQRT2 : 1.0f;
                    float a2 = !v ? ISQRT2 : 1.0f;

                    /* Scale according to normalizing function */
                    dct *= a1*a2/4.0f;

                    /* Quantize */
                    out_data[(y+v)*width+(x+u)] = (int16_t)(floor(0.5f + dct / (float)(quantization[v*8+u])));
                }
            }
        }
    }
}

static void encode(yuv_t *image, dct_t* out)
{
    /* DCT and Quantization */
    dct_quantize(image->Y, width, height, out->Ydct, ypw, yph, yquanttbl);
    dct_quantize(image->U, (width*UX/YX), (height*UY/YY), out->Udct, upw, uph, uquanttbl);
    dct_quantize(image->V, (width*VX/YX), (height*VY/YY), out->Vdct, vpw, vph, vquanttbl);

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

    if( argc == 1 )
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

    if( optind >= argc )
    {
        fprintf(stderr, "Error getting program options, try --help.\n");
        exit(EXIT_FAILURE);
    }

    outfile = fopen(output_file, "wb");
    if( outfile == NULL )
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

    input_file = argv[optind];

    if (limit_numframes)
    {
        printf("Limited to %d frames.\n", limit_numframes);
    }

    FILE *infile = fopen(input_file, "rb");

    if( infile == NULL )
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    /* Allocate data before read and encode loop */
    yuv_t *image = (yuv_t*)malloc(sizeof(yuv_t));
    image->Y = (uint8_t*)malloc(width*height);
    image->U = (uint8_t*)malloc(width*height);
    image->V = (uint8_t*)malloc(width*height);

    cudaError_t err;
    err = cudaMallocManaged( &image->Y, width*height ); fatality_test( err, "Failed to allocate image plane." );
    err = cudaMallocManaged( &image->U, width*height ); fatality_test( err, "Failed to allocate image plane." );
    err = cudaMallocManaged( &image->V, width*height ); fatality_test( err, "Failed to allocate image plane." );

    dct_t *out = (dct_t*)malloc(sizeof(dct_t));
    err = cudaMallocManaged( &out->Ydct, yph*ypw*(sizeof(*out->Ydct)) ); fatality_test( err, "Failed to allocate dct plane." );
    err = cudaMallocManaged( &out->Udct, uph*upw*(sizeof(*out->Udct)) ); fatality_test( err, "Failed to allocate dct plane." );
    err = cudaMallocManaged( &out->Vdct, vph*vpw*(sizeof(*out->Vdct)) ); fatality_test( err, "Failed to allocate dct plane." );

    /* Encode input frames */
    int numframes = 0;
    while( !feof(infile) )
    {
        image = read_yuv(infile, image);

        if (!image) {
            break;
        }

        printf("Encoding frame %d, ", numframes);

        /* Call encode with pointer both the the YUV frame and DCT output */
        encode(image, out);

        printf("Done!\n");

        ++numframes;
        if( limit_numframes && numframes >= limit_numframes ) break;
    }

    /* Free both image read from file and endoded image */
    cudaFree(image->Y);
    cudaFree(image->U);
    cudaFree(image->V);
    free(image);

    cudaFree(out->Ydct);
    cudaFree(out->Udct);
    cudaFree(out->Vdct);
    free(out);

    fclose( outfile );
    fclose( infile );

    exit( EXIT_SUCCESS );
}
