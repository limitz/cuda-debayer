#include <gpu.h>
#include <image.h>
#include <display.h>

#define SYM(v, r) (min(max((int)(v), -(int)(v)), 2*(int)(r)-(int)(v)))
#define RC(type, var, pitch, x, y, width, height) ((type*)(((uint8_t*)(var)) + SYM((y),(height)) * (pitch)) + SYM((x),(width)))
#define IS_R(x,y) (~(x|y)&1)
#define IS_G(x,y) ((x^y)&1)
#define IS_B(x,y) (x&y&1)

__global__ 
void f_conv_f32c3_f32c3(float* k, size_t c, size_t r, float3* out, size_t pitch_out, float3* in, size_t pitch_in, size_t width, size_t height, float factor)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	
	auto s = View2DSym<float3>(in, pitch_in, x, y, width, height);
	auto d = View2DSym<float3>(out, pitch_out, x, y, width, height);

	d(0,0) =  make_float3(0);
	#pragma unroll
	for (int i=0; i<c; i++)
	{
		#pragma unroll
		for (int j=0; j<r; j++, k++)
		{
			d(0,0).x += factor * s(i-c/2,j-r/2).x * *k;
			d(0,0).y += factor * s(i-c/2,j-r/2).y * *k;
			d(0,0).z += factor * s(i-c/2,j-r/2).z * *k;
		}
	}
}

__global__ 
void f_conv_f32c3_u8c3(float* k, size_t c, size_t r, float3* out, size_t pitch_out, uchar3* in, size_t pitch_in, size_t width, size_t height, float factor)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	
	auto s = View2DSym<uchar3>(in, pitch_in, x, y, width, height);
	auto d = View2DSym<float3>(out, pitch_out, x, y, width, height);

	factor *= 1.0/255.0;

	d(0,0) =  make_float3(0,0,0);
	#pragma unroll
	for (int i=0; i<c; i++)
	{
		#pragma unroll
		for (int j=0; j<r; j++, k++)
		{
			d(0,0).x += factor * s(i-c/2,j-r/2).x * *k;
			d(0,0).y += factor * s(i-c/2,j-r/2).y * *k;
			d(0,0).z += factor * s(i-c/2,j-r/2).z * *k;
		}
	}
}
__global__
void f_map(float3* out, size_t pitch_out, size_t width, size_t height, float3 r, float3 g, float3 b, float3 offset)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	
	auto s = View2DSym<float3>(out, pitch_out, x-2, y-2, width, height);
	s(0,0) = r * s(0,0).x + g * s(0,0).y + b * s(0,0).z + offset;
}

void conv(Image* out, Image* in, float* kernel, size_t cols, size_t rows, float factor, cudaStream_t stream)
{
	float *k;
	int rc;

	rc = cudaMalloc(&k, rows * cols * sizeof(float));
	if (cudaSuccess != rc) throw "Unable to allocate space for kernel";

	rc = cudaMemcpyAsync(k, kernel, rows * cols * sizeof(float), cudaMemcpyHostToDevice, stream);
	if (cudaSuccess != rc) throw "Unable to copy kernel to device";

	if (!isnormal(factor))
	{
		for (int i = 0; i < rows * cols; i++) factor += kernel[i];
		factor = 1.0 / factor;
	}

	dim3 blockSize = { 16, 16 };
	dim3 gridSize = { 
		((int)in->width  + blockSize.x - 1) / blockSize.x, 
		((int)in->height + blockSize.y - 1) / blockSize.y
	};

	if (in->type == Image::Type::lab)
	{
		f_conv_f32c3_f32c3 <<< gridSize, blockSize, 0, stream >>> (
			k, cols, rows,
			(float3*) out->mem.device.data, out->mem.device.pitch,
			(float3*) in->mem.device.data, in->mem.device.pitch,
			in->width, in->height, factor);

	}
	else if (in->type == Image::Type::ppm)
	{
		f_conv_f32c3_u8c3 <<< gridSize, blockSize, 0, stream >>> (
			k, cols, rows,
			(float3*) out->mem.device.data, out->mem.device.pitch,
			(uchar3*) in->mem.device.data, in->mem.device.pitch,
			in->width, in->height, factor);
	}
	cudaStreamSynchronize(stream);
	cudaFree(k);
}

void channelmap(Image* img, const float map[9], const float offsets[3], cudaStream_t stream)
{
	dim3 blockSize = { 16, 16 };
	dim3 gridSize = { 
		((int)img->width  + blockSize.x - 1) / blockSize.x, 
		((int)img->height + blockSize.y - 1) / blockSize.y
	};

	f_map <<< gridSize, blockSize, 0, stream >>> (
		(float3*)img->mem.device.data, img->mem.device.pitch,
		img->width, img->height,
		make_float3(map[0], map[1], map[2]),
		make_float3(map[3], map[4], map[5]),
		make_float3(map[6], map[7], map[8]),
		make_float3(offsets[0], offsets[1], offsets[2]));
}

__global__
void f_pgm8(float4* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height, size_t scale, int dx, int dy)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	
	float px = *RC(uint8_t, in, pitch_in, x/scale+dx, y/scale+dy, width, height)/255.0;
	*RC(float4, out, pitch_out, x, y, width, height) = make_float4(px, px, px, 1.0);
}

__global__
void f_ppm8(float4* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height, size_t scale, int dx, int dy)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	uchar3 p = *RC(uchar3, in, pitch_in, x/scale+dx, y/scale+dy, width, height);
	*RC(float4, out, pitch_out, x, y, width, height) = make_float4(p.x/255.0f, p.y/255.0f, p.z/255.0f, 1.0);
}

__global__
void f_lab(float4* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height, size_t scale, int dx, int dy, int type)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	float3 p = clamp(*RC(float3, in, pitch_in, x/scale+dx, y/scale+dy, width, height)/128, -1.0f, 1.0f);
	float sat = clamp(sqrt(p.y * p.y + p.z * p.z), 0.0f, 1.0f);
	*RC(float4, out, pitch_out, x, y, width, height) = make_float4(
			type == 0 ? p.x : type == 1 ?  p.x + p.y - p.z/2 : p.x/10 + sat,
			type == 0 ? p.x : type == 1 ? p.x + -p.y - p.z/2 : p.x/10 + sat,
			type == 0 ? p.x : type == 1 ? p.x + -p.z : p.x/10 + sat,1.0);
}

__global__
void f_blend_f32c3(
		float3* out, size_t pitch_out, 
		float3* a, size_t pitch_a, 
		float3* b, size_t pitch_b, 
		float3* mask, size_t pitch_mask, 
		size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	float3 f = clamp(*RC(float3, mask, pitch_mask, x, y, width, height), 0.0f, 1.0f);
	float3 va = *RC(float3, a, pitch_a, x, y, width, height);
	float3 vb = *RC(float3, b, pitch_b, x, y, width, height);
	float3 ia = make_float3(
			f.x * va.x,
			f.y * va.y,
			f.z * va.z);
	float3 ib = make_float3(
			(1-f.x) * vb.x,
			(1-f.y) * vb.y,
			(1-f.z) * vb.z);
	float3 blend = ia + ib;

	*RC(float3, out, pitch_out, x, y, width, height) = blend;
}

void blend(Image* out, Image* mask, Image* a, Image* b, cudaStream_t stream)
{
	dim3 blockSize = { 16, 16 };
	dim3 gridSize = { 
		((int)out->width  + blockSize.x - 1) / blockSize.x, 
		((int)out->height + blockSize.y - 1) / blockSize.y
	};

	f_blend_f32c3 <<< gridSize, blockSize, 0, stream >>> (
		(float3*) out->mem.device.data, out->mem.device.pitch,
		(float3*) a->mem.device.data, a->mem.device.pitch,
		(float3*) b->mem.device.data, b->mem.device.pitch,
		(float3*) mask->mem.device.data, mask->mem.device.pitch,
		out->width, out->height);
}

int smToCores(int major, int minor)
{
	switch ((major << 4) | minor)
	{
		case (9999 << 4 | 9999):
			return 1;
		case 0x30:
		case 0x32:
		case 0x35:
		case 0x37:
			return 192;
		case 0x50:
		case 0x52:
		case 0x53:
			return 128;
		case 0x60:
			return 64;
		case 0x61:
		case 0x62:
			return 128;
		case 0x70:
		case 0x72:
		case 0x75:
			return 64;
		case 0x80:
		case 0x86:
			return 64;
		default:
			return 0;
	};
}

void display(CudaDisplay* display, Image* img, float scale, size_t dx, size_t dy, int type, cudaStream_t stream)
{
	dim3 blockSize = { 16,16 };
	dim3 gridSize = { 
		((int)display->CUDA.frame.width  + blockSize.x - 1) / blockSize.x,
		((int)display->CUDA.frame.height + blockSize.y - 1) / blockSize.y };

	switch (img->type)
	{
		case Image::Type::lab:
			f_lab <<< gridSize, blockSize, 0, stream >>> (
				display->CUDA.frame.data,
				display->CUDA.frame.pitch,
				img->mem.device.data,
				img->mem.device.pitch,
				img->width,
				img->height,
				scale,
				dx * scale, dy * scale,
				type);
			break;
		case Image::Type::pgm:
			f_pgm8 <<< gridSize, blockSize, 0, stream >>> (
				display->CUDA.frame.data,
				display->CUDA.frame.pitch,
				img->mem.device.data,
				img->mem.device.pitch,
				img->width,
				img->height,
				scale,
				dx * scale, dy * scale);
			break;
		case Image::Type::ppm:
			f_ppm8 <<< gridSize, blockSize, 0, stream >>> (
				display->CUDA.frame.data,
				display->CUDA.frame.pitch,
				img->mem.device.data,
				img->mem.device.pitch,
				img->width,
				img->height,
				scale,
				dx * scale, dy * scale);
			break;
		default:
			throw "Unexpected image type";
	}
}

void selectGPU()
{
	int rc;
	int maxId = -1;
	uint16_t maxScore = 0;
	int count = 0;
	cudaDeviceProp prop;

	rc = cudaGetDeviceCount(&count);
	if (cudaSuccess != rc) throw "cudaGetDeviceCount error";
	if (count == 0) throw "No suitable cuda device found";

	for (int id = 0; id < count; id++)
	{
		rc = cudaGetDeviceProperties(&prop, id);
		if (cudaSuccess != rc) throw "Unable to get device properties";
		if (prop.computeMode == cudaComputeModeProhibited) 
		{
			printf("GPU %d: PROHIBITED\n", id);
			continue;
		}
		int sm_per_multiproc = smToCores(prop.major, prop.minor);
		
		printf("GPU %d: \"%s\"\n", id, prop.name);
		printf(" - Compute capability: %d.%d\n", prop.major, prop.minor);
		printf(" - Multiprocessors:    %d\n", prop.multiProcessorCount);
		printf(" - SMs per processor:  %d\n", sm_per_multiproc);
		printf(" - Clock rate:         %d\n", prop.clockRate);

		uint64_t score =(uint64_t) prop.multiProcessorCount * sm_per_multiproc * prop.clockRate;
		if (score > maxScore) 
		{
			maxId = id;
			maxScore = score;
		}
	}

	if (maxId < 0) throw "All cuda devices prohibited";

	rc = cudaSetDevice(maxId);
	if (cudaSuccess != rc) throw "Unable to set cuda device";

	rc = cudaGetDeviceProperties(&prop, maxId);
	if (cudaSuccess != rc) throw "Unable to get device properties";

	printf("\nSelected GPU %d: \"%s\" with compute capability %d.%d\n\n", 
		maxId, prop.name, prop.major, prop.minor);
}
