#pragma once

extern __device__ __constant__ float cosv[8][8];

#define COSUV(i, j, k, l) ((float) (cosv[k][i] * cosv[l][j]))

