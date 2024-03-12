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

static uint32_t bit_buffer = 0;
static uint32_t bit_buffer_width = 0;

static void put_byte(int byte)
{
    int status = fputc(byte, outfile);

    if( status == EOF )
    {
        fprintf(stderr, "Error writing byte\n");
        exit(EXIT_FAILURE);
    }
}

static void put_bytes(const void* data, unsigned int len)
{
    int n = fwrite(data, 1, len, outfile);

    if( n != len )
    {
        fprintf(stderr, "Error writing bytes\n");
        exit(-1);
    }
}

void write_SOI()
{
    put_byte(0xff);
    put_byte(0xd8);
}

void write_DQT()
{
    int16_t size = 2 + (3 * 65);

    put_byte(0xff);
    put_byte(0xdb);

    put_byte(size >> 8);
    put_byte(size & 0xff);

    put_byte(0);
    put_bytes(yquanttbl, 64);
    put_byte(1);
    put_bytes(uquanttbl, 64);
    put_byte(2);
    put_bytes(vquanttbl, 64);
}

void write_SOF0()
{
    int16_t size = 8 + 3 * COLOR_COMPONENTS;

    /* Header marker */
    put_byte(0xff);
    put_byte(0xc0);

    /* Size of header */
    put_byte(size >> 8);
    put_byte(size & 0xff);

    /* Precision */
    put_byte(8);

    /* Width and height */
    put_byte(height >> 8);
    put_byte(height & 0xff);
    put_byte(width >> 8);
    put_byte(width & 0xff);

    put_byte(COLOR_COMPONENTS);

    put_byte(1); /* Component id */
    put_byte(0x22); /* hor | ver sampling factor FIXME Y(2,2), U(1,1), V(1,1) */
    put_byte(0); /* Quant. tbl. id */

    put_byte(2); /* Component id */
    put_byte(0x11); /* hor | ver sampling factor */
    put_byte(1); /* Quant. tbl. id */

    put_byte(3); /* Component id */
    put_byte(0x11); /* hor | ver sampling factor */
    put_byte(2); /* Quant. tbl. id */
}

static void write_DHT_HTS(uint8_t id, uint8_t *numlength, uint8_t* data)
{
    /* Find out how many codes we are to write */
    int i;
    int n = 0;
    for(i = 0; i < 16; ++i)
        n += numlength[i];

    put_byte(id);
    put_bytes(numlength, 16);
    put_bytes(data, n);
}

void write_DHT()
{
    int16_t size = 0x01A2; /* 2 + n*(17+mi); */

    /* Define Huffman Table marker */
    put_byte(0xff);
    put_byte(0xc4);

    /* Length of segment */
    put_byte(size >> 8);
    put_byte(size & 0xff);

    /* Write the four huffman table specifications */
    write_DHT_HTS(0x00, DCVLC_num_by_length[0], DCVLC_data[0]); /* DC table 0 */
    write_DHT_HTS(0x01, DCVLC_num_by_length[1], DCVLC_data[1]); /* DC table 1 */
    write_DHT_HTS(0x10, ACVLC_num_by_length[0], ACVLC_data[0]); /* AC table 0 */
    write_DHT_HTS(0x11, ACVLC_num_by_length[1], ACVLC_data[1]); /* AC table 1 */
}

void write_SOS()
{
    int16_t size = 6 + 2 * COLOR_COMPONENTS;

    put_byte(0xff);
    put_byte(0xda);

    put_byte(size >> 8);
    put_byte(size & 0xff);

    put_byte(COLOR_COMPONENTS);

    put_byte(1); /* Component id */
    put_byte(0x00); /* DC | AC huff tbl */
    put_byte(2); /* Component id */
    put_byte(0x11); /* DC | AC huff tbl */
    put_byte(3); /* Component id */
    put_byte(0x11); /* DC | AC huff tbl */
    put_byte(0); /* ss, first AC */
    put_byte(63); /* se, last AC */
    put_byte(0); /* ah | al */
}

void write_EOI()
{
    put_byte(0xff);
    put_byte(0xd9);
}

static inline uint8_t bit_width(int16_t i)
{
    /* replacing ILOG2 with M_LOG2E from math.h */
    return (uint8_t) (ceil((log(abs(i)+1))*M_LOG2E));
}

/**
 * Adds a bit to the bitBuffer. A call to Flush() is needed
 * in order to write any remainding bits in the buffer before
 * writing using another function.
 */
static void put_bits(int16_t bits, uint8_t n)
{
    assert(n <= 24  && "Error writing bit");

    if(n == 0)
        return;

    bit_buffer <<= n;
    bit_buffer |= bits & ((1 << n) - 1);
    bit_buffer_width += n;

    while(bit_buffer_width >= 8)
    {
        uint8_t c = (uint8_t)(bit_buffer >> (bit_buffer_width - 8));
        put_byte(c);
        if( c == 0xff )
            put_byte(0);

        bit_buffer_width -= 8;
    }
}

/**
 * Flushes the bitBuffer by writing zeroes to fill a full byte
 */
static void flush()
{
    if( bit_buffer > 0 )
    {
        uint8_t c = bit_buffer << (8 - bit_buffer_width);
        put_byte(c);
        if( c == 0xff )
            put_byte(0);
    }

    bit_buffer = 0;
    bit_buffer_width = 0;
}

static void write_block( int16_t *in_data, uint32_t width, uint32_t height,
                         uint32_t uoffset, uint32_t voffset, int16_t* prev_DC,
                         int32_t cc)
{
    uint32_t zigzag, i, j;

    static int16_t block[64];
    int32_t num_ac = 0;

    /* ZigZag */
    for( zigzag = 0; zigzag < 64; zigzag++ )
    {
        uint8_t u = zigzag_U[zigzag];
        uint8_t v = zigzag_V[zigzag];
        block[zigzag] = in_data[(voffset+v)*width+(uoffset+u)];
    }

    /* Calculate DC component, and write to stream */
    int16_t dc = block[0] - *prev_DC;
    *prev_DC = block[0];
    uint8_t size = bit_width(dc);
    put_bits(DCVLC[cc][size],DCVLC_Size[cc][size]);
    if( dc < 0 )
        dc = dc - 1;

    put_bits(dc, size);

    /* find the last nonzero entry of the ac-coefficients */
    for( j = 64; j > 1 && !block[j-1]; j-- )
        ;

    /* Put the nonzero ac-coefficients */
    for( i = 1; i < j; i++ )
    {
        int16_t ac = block[i];
        if( ac == 0 )
        {
            if( ++num_ac == 16 )
            {
                put_bits(ACVLC[cc][15][0], ACVLC_Size[cc][15][0]);
                num_ac = 0;
            }
        }
        else
        {
            uint8_t size = bit_width(ac);
            put_bits(ACVLC[cc][num_ac][size], ACVLC_Size[cc][num_ac][size]);

            if( ac < 0 )
                --ac;
            put_bits(ac, size);
            num_ac = 0;
        }
    }

    /* Put end of block marker */
    if( j < 64 )
        put_bits(ACVLC[cc][0][0], ACVLC_Size[cc][0][0]);
}

static void write_interleaved_data_MCU( int16_t *dct, uint32_t wi, uint32_t he,
                                        uint32_t h, uint32_t v, uint32_t x,
                                        uint32_t y, int16_t *prev_DC, int32_t cc)
{
    uint32_t i, j, ii, jj;
    for( j = y*v*8; j < (y+1)*v*8; j += 8 )
    {
        jj = he-8;
        jj = MIN(j, jj);

        for( i = x*h*8; i < (x+1)*h*8; i += 8 )
        {
            ii = wi-8;
            ii = MIN(i, ii);

            write_block(dct, wi, he, ii, jj, prev_DC, cc);
        }
    }
}

void write_interleaved_data(dct_t *out)
{
    int16_t prev_DC[3] = {0, 0, 0};
    uint32_t u, v;

    /* Set up which huffman tables we want to use */
    int32_t yhtbl = 0;
    int32_t uhtbl = 1;
    int32_t vhtbl = 1;

    /* Find the number of MCU's for the intensity */
    uint32_t ublocks = (uint32_t) (ceil(ypw/(float)(8.0f*YX)));
    uint32_t vblocks = (uint32_t) (ceil(yph/(float)(8.0f*YY)));

    /* Write the MCU's interleaved */
    for( v = 0; v < vblocks; ++v )
    {
        for( u = 0; u < ublocks; ++u )
        {
            write_interleaved_data_MCU(out->Ydct, ypw, yph, YX, YY, u, v, &prev_DC[0], yhtbl);
            write_interleaved_data_MCU(out->Udct, upw, uph, UX, UY, u, v, &prev_DC[1], uhtbl);
            write_interleaved_data_MCU(out->Vdct, vpw, vph, VX, VY, u, v, &prev_DC[2], vhtbl);
        }
    }

    flush();
}

