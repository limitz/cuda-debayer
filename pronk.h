#pragma once
#include <cuda_runtime.h>
#include <operators.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <image.h>

class PronkFilter
{
public:
	PronkFilter();
	~PronkFilter();

	Image* source;
	Image* destination;
	dim3   blockSize;
	
	void   run(cudaStream_t stream);
};
