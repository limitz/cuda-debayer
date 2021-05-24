#include <sobel.h>
#include <view.h>

__constant__ __device__ float sobel_kernel_x[9];
__constant__ __device__ float sobel_kernel_y[9];

__global__
void f_sobel(float3* dst, size_t dst_pitch, uchar3* src, size_t src_pitch, size_t width, size_t height, bool avg, float power)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	auto d = View2DSym<float3>(dst, dst_pitch, x, y, width, height);
	auto s = View2DSym<uchar3>(src, src_pitch, x, y, width, height);

	float3 Lx = make_float3(0,0,0);
	float3 Ly = make_float3(0,0,0);

	float* Kx = sobel_kernel_x;
	float* Ky = sobel_kernel_y;

	#pragma unroll
	for (int r = -1; r <= 1; r++)
	{
		#pragma unroll
		for (int c = -1; c <= 1; c++, Kx++, Ky++)
		{
			float kx = Kx[0];
			float ky = Ky[0];
			float3 v = make_float3(s(c,r)) / 255.0;
			Lx += kx * v;
			Ly += ky * v;
		}
	}
	if (avg)
	{
		float avgx = (Lx.x + Lx.y + Lx.z)/3.0f;
		float avgy = (Ly.x + Ly.y + Ly.z)/3.0f;
		float gradient = clamp(pow(avgx*avgx + avgy*avgy, power), 0.0f, 1.0f);
		d(0,0) = make_float3(gradient);
	}
	else
	{
		float3 gradient = make_float3(
				pow(Lx.x*Lx.x + Ly.x*Ly.x, power),
				pow(Lx.y*Lx.y + Ly.y*Ly.y, power),
				pow(Lx.z*Lx.z + Ly.z*Ly.z, power));

		d(0,0) = clamp(gradient, 0.0f, 1.0f);	
	}
}

SobelFilter::SobelFilter()
{
	float kx[9] = {
		+1, 0, -1,
		+2, 0, -2,
		+1, 0, -1,
	};

	float ky[9] = {
		+1, +2, +1,
		 0,  0,  0,
		-1, -2, -1,
	};

	blockSize = { 16,16 };
	int rc;

	avgChannels = false;

	rc = cudaMemcpyToSymbol(sobel_kernel_x, &kx, 9*sizeof(float), 0, cudaMemcpyHostToDevice);
	if (cudaSuccess != rc) throw "Unable to copy kernel x to device";

	rc = cudaMemcpyToSymbol(sobel_kernel_y, &ky, 9*sizeof(float), 0, cudaMemcpyHostToDevice);
	if (cudaSuccess != rc) throw "Unable to copy kernel y to device";

	cudaDeviceSynchronize();
}

SobelFilter::~SobelFilter()
{
}

void SobelFilter::run(cudaStream_t stream)
{
	
	if (!source) throw "Source not set";
	if (!destination) throw "Destination not set";
	if (source->width != destination->width || source->height != destination->height) 
		throw "Images not the same size";
	
	source->printInfo();

	dim3 gridSize = {
		(((int)source->width  + blockSize.x - 1) / blockSize.x),
		(((int)source->height + blockSize.y - 1) / blockSize.y)
	};

	f_sobel <<< gridSize, blockSize, 0, stream >>> (
		(float3*)destination->mem.device.data, destination->mem.device.pitch,
		(uchar3*)source->mem.device.data, source->mem.device.pitch,
		source->width, source->height, avgChannels, power);
}
