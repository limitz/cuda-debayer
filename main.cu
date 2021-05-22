#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/stat.h>

#define WIDTH 1920
#define HEIGHT 1080

#include <display.h>
#include <pthread.h>
#include <math.h>
#include <operators.h>
#include <image.h>

#ifndef TITLE
#define TITLE "CUDA DEBAYER DEMO"
#endif

__global__
void f_test(float4* out, int pitch_out, int width, int height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	out[y * pitch_out / sizeof(float4) + x] = make_float4(
			(float) x / width, 
			(float) y / height, 
			0, 1);
}

__global__
void f_pgm8(float4* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	uchar1 p = ((uchar1*)(((uint8_t*) in) + pitch_in * y))[x];
	out[y * pitch_out / sizeof(float4) + x] = make_float4(
			p.x / 255.0f,
			p.x / 255.0f,
			p.x / 255.0f,
			1.0);
}

__global__
void f_ppm8(float4* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	uchar3 p = ((uchar3*)(((uint8_t*) in) + pitch_in * y))[x];
	out[y * pitch_out / sizeof(float4) + x] = make_float4(
			p.x / 255.0f,
			p.y / 255.0f,
			p.z / 255.0f,
			1.0);
}

__global__
void f_ppm8_bayer_pgm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	uchar3 m  = make_uchar3(~(x | y) & 1, (x ^ y) & 1,x & y & 1); 
	uchar3 p  = ((uchar3*)(((uint8_t*) in)  + pitch_in  * y))[x];
	uchar1* o = ((uchar1*)(((uint8_t*) out) + pitch_out * y)) + x;
	*o = make_uchar1(
			m.x * p.x +
			m.y * p.y +
			m.z * p.z);
}

__global__
void f_pgm8_debayer_bilinear_ppm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x < 1 || y < 1 || x >= width-1 || y >= height-1) return;

	uint8_t r,g,b;
	uint8_t* s1 = (((uint8_t*) in)  + pitch_in  * y) + x;
	uint8_t* s2 = s1 + pitch_in;
	uchar3*  o1 = ((uchar3*)(((uint8_t*)out) + pitch_out * (y+0))) + x;	
	uchar3*  o2 = ((uchar3*)(((uint8_t*)out) + pitch_out * (y+1))) + x;	
	
	// red
	o1[0] = make_uchar3(
		(s1[0]),
		(s1[-1] + s1[+1] + s1[-pitch_in] + s1[pitch_in]  ) / 4,
		(s1[-1-pitch_in] + s1[1-pitch_in] + s1[pitch_in-1] + s1[pitch_in+1]) / 4
	);
	
	// red green
	o1[1] = make_uchar3(
		(s1[0] + s1[2]) / 2,
		s1[1],
		(s1[1-pitch_in] + s1[1+pitch_in]) / 2
	);

	// blue
	o2[1] = make_uchar3(
		(s2[-pitch_in] + s2[+2-pitch_in] + s2[pitch_in] + s2[pitch_in+2]) / 4,
		(s2[0] + s2[+2] + s2[1-pitch_in] + s2[1+pitch_in]) / 4,
		s2[1]
	);
	
	// blue green
	o2[0] = make_uchar3(
		(s2[-pitch_in] + s2[pitch_in]) / 2,
		s2[0],
		(s2[-1] + s2[1]) / 2
	);
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

int main(int /*argc*/, char** /*argv*/)
{
	int rc;
	cudaStream_t stream = 0;

	try 
	{
		printf("Selecting the best GPU\n");
		selectGPU();
		
		dim3 blockSize = { 16, 16 };
		dim3 gridSize = { 
			(WIDTH  + blockSize.x - 1) / blockSize.x, 
			(HEIGHT + blockSize.y - 1) / blockSize.y 
		}; 
		dim3 gridSizeQ = { 
			(WIDTH/2  + blockSize.x - 1) / blockSize.x, 
			(HEIGHT/2 + blockSize.y - 1) / blockSize.y 
		}; 

		rc = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		if (cudaSuccess != rc) throw "Unable to create CUDA stream";

		auto original = Image::load("sheep.ppm");
		original->printInfo();
		original->copyToDevice(stream);

		auto bayer = Image::create(Image::Type::pgm8, original->width, original->height);
		f_ppm8_bayer_pgm8<<<gridSize, blockSize, 0, stream>>>(
				bayer->data.device,
				bayer->pitch,
				original->data.device,
				original->pitch,
				original->width,
				original->height
		);

		auto debayer = Image::create(Image::Type::ppm8, original->width, original->height);
		f_pgm8_debayer_bilinear_ppm8<<<gridSizeQ, blockSize, 0, stream>>>(
				debayer->data.device,
				debayer->pitch,
				bayer->data.device,
				bayer->pitch,
				bayer->width,
				bayer->height
		);

		printf("Creating screen\n");
		CudaDisplay display(TITLE, WIDTH, HEIGHT); 
		cudaDeviceSynchronize();
		
		display.cudaMap(stream);
		while (true)
		{
#if 0
			f_pgm8<<<gridSize, blockSize, 0, stream>>>(
				display.CUDA.frame.data,
				display.CUDA.frame.pitch,
				bayer->data.device,
				bayer->pitch,
				bayer->width,
				bayer->height
			);
#else

			f_ppm8<<<gridSize, blockSize, 0, stream>>>(
				display.CUDA.frame.data,
				display.CUDA.frame.pitch,
				debayer->data.device,
				debayer->pitch,
				debayer->width,
				debayer->height
			);
#endif

			// Draw the pixelbuffer on screen
			display.cudaFinish(stream);
			display.render(stream);
		
			rc = cudaGetLastError();
			if (cudaSuccess != rc) throw "CUDA ERROR";

			// check escape pressed
			if (display.events()) 
			{
				display.cudaUnmap(stream);
				cudaStreamDestroy(stream);
				return 0;
			}
			usleep(1000);
		}
	}
	catch (const char* &ex)
	{
		fprintf(stderr, "ERROR: %s\n", ex);
		fflush(stderr);
	 	return 1;
	}
	return 0;
}
