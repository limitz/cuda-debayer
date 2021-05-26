#include <malvar.h>

__constant__ __device__ int32_t malvar[100];
void MalvarFilter::setup(cudaStream_t stream)
{
	int32_t pmalvar[100] = 
	{
		 0,  0, -2,  0,  0,
		 0,  0,  4,  0,  0,
		-2,  4,  8,  4, -2,
		 0,  0,  4,  0,  0,
		 0,  0, -2,  0,  0,

		 0,  0,  1,  0,  0,
		 0, -2,  0, -2,  0,
		-2,  8, 10,  8, -2,
		 0, -2,  0, -2,  0,
		 0,  0,  1,  0,  0,

		 0,  0, -2,  0,  0,
		 0, -2,  8, -2,  0,
		 1,  0, 10,  0,  1,
		 0, -2,  8, -2,  0,
		 0,  0, -2,  0,  0,
		
		 0,  0, -3,  0,  0,
		 0,  4,  0,  4,  0,
		-3,  0, 12,  0, -3,
		 0,  4,  0,  4,  0,
		 0,  0, -3,  0,  0,
	};

	int rc = cudaMemcpyToSymbolAsync(
			malvar, &pmalvar,
			100*sizeof(int32_t), 0,
			cudaMemcpyHostToDevice,
			stream);

	if (cudaSuccess != rc) throw "Unable to copy malvar kernels";
}

__global__
void f_malvar(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x >= width || y >= height) return;

	auto d = View2DSym<uchar3>(out, pitch_out, x, y, width, height);
	auto s = View2DSym<uint8_t>(in, pitch_in,  x, y, width, height);
	
	int3 trr = make_int3(s(0,0), 0, 0), trg = make_int3(0, s(1,0), 0),
	     tbg = make_int3(0, s(0,1), 0), tbb = make_int3(0, 0, s(1,1));

	#pragma unroll
	for (int r=-2, *m=malvar; r<3; r++)
	{
		#pragma unroll
		for (int c=-2; c<3; c++, m++)
		{
			trr.y += m[ 0] * s(c+0, r+0), trr.z += m[75] * s(c+0, r+0);
			trg.x += m[25] * s(c+1, r+0), trg.z += m[50] * s(c+1, r+0);
			tbg.x += m[50] * s(c+0, r+1), tbg.z += m[25] * s(c+0, r+1);
			tbb.x += m[75] * s(c+1, r+1), tbb.y += m[ 0] * s(c+1, r+1);
		}
	}

	d(0,0) = make_uchar3(trr.x, clamp(trr.y, 0, 0xFF0) >> 4, clamp(trr.z, 0, 0xFF0) >> 4);
	d(1,0) = make_uchar3(clamp(trg.x, 0, 0xFF0) >> 4, trg.y, clamp(trg.z, 0, 0xFF0) >> 4);
	d(0,1) = make_uchar3(clamp(tbg.x, 0, 0xFF0) >> 4, tbg.y, clamp(tbg.z, 0, 0xFF0) >> 4);
	d(1,1) = make_uchar3(clamp(tbb.x, 0 ,0xFF0) >> 4, clamp(tbb.y, 0, 0xFF0) >> 4, tbb.z);
}

void MalvarFilter::run(cudaStream_t stream)
{
	Filter::run(stream);

	f_malvar <<< gridSizeQ, blockSize, 0, stream >>> (
		destination->mem.device.data, destination->mem.device.pitch,
		source->mem.device.data, source->mem.device.pitch,
		source->width, source->height);
}
