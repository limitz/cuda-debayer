#include <cuda_runtime.h>
#include <operators.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <image.h>

class SobelFilter
{
public:
	SobelFilter();
	~SobelFilter();

	Image* source;
	Image* destination;
	dim3   blockSize;

	void run(cudaStream_t stream);
};
