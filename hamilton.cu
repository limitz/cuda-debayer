#include <hamilton.h>

__global__
void f_hamilton_gg(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height, int threshold)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x >= width || y >= height) return;
	
	auto d = View2DSym<uchar3>(out, pitch_out, x, y, width, height);
	auto s = View2DSym<uint8_t>(in, pitch_in,  x, y, width, height);
	
	// greens
	d(0,0) = make_uchar3(0, 0, 0);
	d(1,0) = make_uchar3(0, s(1,0), 0);
	d(0,1) = make_uchar3(0, s(0,1), 0);
	d(1,1) = make_uchar3(0, 0, 0);

	// greens at red / blue positions
	#pragma unroll
	for (int i=0; i<2; i++)
	{
		float green;
		int dh = abs(s(i-1,i)-s(i+1,i))+abs(2*s(i,i)-s(i+2,i)-s(i-2,i));
		int dv = abs(s(i,i-1)-s(i,i+1))+abs(2*s(i,i)-s(i,i+2)-s(i,i-2));
		
		if (dh > dv+threshold) 
			green = (s(i,i-1)+s(i,i+1))*0.5f+(2*s(i,i)-s(i,i-2)-s(i,i+2))*0.25f;
		else if (dv > dh+threshold) 
			green = (s(i-1,i)+s(i+1,i))*0.5f+(2*s(i,i)-s(i-2,i)-s(i+2,i))*0.25f;
		else
			green = (s(i,i-1)+s(i,i+1)+s(i-1,i)+s(i+1,i))*0.25f 
			      + (4*s(i,i)-s(i,i-2)-s(i,i+2)-s(i-2,i)-s(i+2,i))*0.125f;

		d(i,i).y = (uint8_t) clamp(green, 0.0f, 255.0f);
	}
}

__global__
void f_hamilton_rb(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height, int threshold)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x >= width || y >= height) return;
	
	auto d = View2DSym<uchar3>(out, pitch_out, x, y, width, height);
	auto s = View2DSym<uint8_t>(in, pitch_in,  x, y, width, height);
	
	d(0,0).x = (s(0,0));
	d(0,0).z = (s(-1,-1)+s(1,-1)+s(-1,1)+s(1,1)) >> 2;
	d(0,1) = make_uchar3(0, s(0,1), 0);
	d(1,1) = make_uchar3(0, 0, 0);

	#pragma unroll
	for (int i=0; i<2; i++)
	{
		float green;
		int dh = abs(s(i-1,i)-s(i+1,i))+abs(2*s(i,i)-s(i+2,i)-s(i-2,i));
		int dv = abs(s(i,i-1)-s(i,i+1))+abs(2*s(i,i)-s(i,i+2)-s(i,i-2));
		
		if (dh > dv+threshold) 
			green = (s(i,i-1)+s(i,i+1))*0.5f+(2*s(i,i)-s(i,i-2)-s(i,i+2))*0.25f;
		else if (dv > dh+threshold) 
			green = (s(i-1,i)+s(i+1,i))*0.5f+(2*s(i,i)-s(i-2,i)-s(i+2,i))*0.25f;
		else
			green = (s(i,i-1)+s(i,i+1)+s(i-1,i)+s(i+1,i))*0.25f 
			      + (4*s(i,i)-s(i,i-2)-s(i,i+2)-s(i-2,i)-s(i+2,i))*0.125f;

		d(i,i).y = (uint8_t) clamp(green, 0.0f, 255.0f);
	}
}

void HamiltonFilter::run(cudaStream_t stream)
{
	Filter::run(stream);

	f_hamilton_gg <<< gridSizeQ, blockSize, 0, stream >>> (
		destination->mem.device.data,
		destination->mem.device.pitch,
		source->mem.device.data,
		source->mem.device.pitch,
		source->width,
		source->height,
		threshold);

	f_hamilton_rb <<< gridSizeQ, blockSize, 0, stream >>> (
		destination->mem.device.data,
		destination->mem.device.pitch,
		source->mem.device.data,
		source->mem.device.pitch,
		source->width,
		source->height,
		threshold);
}
