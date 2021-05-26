#pragma once

__device__ __host__ inline int debayerIsRed(int x, int y)   { return (~(x|y)&1); }
__device__ __host__ inline int debayerIsGreen(int x, int y) { return ((x^y)&1); }
__device__ __host__ inline int debayerIsBlue(int x, int y)  { return (x&y&1); }

