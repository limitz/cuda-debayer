#pragma once
#include <filter.h>
#include <image.h>

__device__ __host__ inline int debayerIsRed(int x, int y)   { return (~(x|y)&1); }
__device__ __host__ inline int debayerIsGreen(int x, int y) { return ((x^y)&1); }
__device__ __host__ inline int debayerIsBlue(int x, int y)  { return (x&y&1); }

class DebayerFilter : public Filter
{
public:
	DebayerFilter() : Filter("Debayer") { }

	static Image* unpack(Image* b,cudaStream_t stream);
	static Image* pack(Image* rgb, cudaStream_t stream);
	virtual void run(cudaStream_t stream) override;
};

class BilinearDebayerFilter : public Filter
{
public:
	BilinearDebayerFilter() : Filter("Bilinear Debayer") { }
	virtual void run(cudaStream_t stream) override;
};
