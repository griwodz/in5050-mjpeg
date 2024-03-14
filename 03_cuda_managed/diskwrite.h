#ifndef DISKWRITE_H
#define DISKWRITE_H

struct dct;
typedef struct dct dct_t;

void write_SOI();
void write_DQT();
void write_SOF0();
void write_DHT();
void write_SOS();
void write_EOI();
void write_interleaved_data(dct_t *out);

#endif
