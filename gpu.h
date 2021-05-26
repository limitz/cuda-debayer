#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/stat.h>
#include <display.h> //move this out
#include <image.h>
#include <view.h>

void conv(Image* out, Image* in, float* kernel, size_t cols, size_t rows, float factor, cudaStream_t stream);
void channelmap(Image* img, const float map[9], const float offsets[3], cudaStream_t stream);
void blend(Image* out, Image* mask, Image* a, Image* b, cudaStream_t stream);
void display(CudaDisplay* display, Image* ppm, float scale, size_t dx, size_t dy, int type, cudaStream_t stream);
void selectGPU();
