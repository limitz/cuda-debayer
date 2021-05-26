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
#include <sobel.h>
#include <hamilton.h>
#include <gunturk.h>
#include <pronk.h>
#include <view.h>

#ifndef TITLE
#define TITLE "CUDA DEBAYER DEMO"
#endif

#define SYM(v, r) (min(max((int)(v), -(int)(v)), 2*(int)(r)-(int)(v)))
#define RC(type, var, pitch, x, y, width, height) ((type*)(((uint8_t*)(var)) + SYM((y),(height)) * (pitch)) + SYM((x),(width)))
#define IS_R(x,y) (~(x|y)&1)
#define IS_G(x,y) ((x^y)&1)
#define IS_B(x,y) (x&y&1)

#define Lab_e 0.008856f
#define Lab_k 903.3f
#define Lab_v 0.0031308
#define Lab_vi 0.04045

__constant__ __device__ float Lab_M[9];
__constant__ __device__ float Lab_Mi[9];
__constant__ __device__ float3 Lab_W;


__global__
void f_cielab_enhance(float3* lab, size_t pitch_in, size_t width, size_t height, float angle, float sat, float bri, float ofs, float da, float db)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	
	float3* px = RC(float3, lab, pitch_in, x, y, width, height);

	px->y = cos(angle)  * px->y + sin(angle) * px->z;
	px->z = -sin(angle) * px->y + cos(angle) * px->z;
	px->x *= bri;
	px->x += ofs;
	px->y *= sat;
	px->z *= sat;
	px->y += da;
	px->z += db;
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
void f_cielab(float4* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height, size_t scale, int dx, int dy, bool l_only, bool s_only)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	float3 p = clamp(*RC(float3, in, pitch_in, x/scale+dx, y/scale+dy, width, height)/100, -1.0f, 1.0f);
	float sat = clamp(sqrt(p.y * p.y + p.z * p.z), 0.0f, 1.0f);
	*RC(float4, out, pitch_out, x, y, width, height) = make_float4(
			l_only ? p.x : s_only ?  p.y + p.z/2 : p.x + p.y + p.z/2,
			l_only ? p.x : s_only ? -p.y + p.z/2 : p.x - p.y + p.z/2,
			l_only ? p.x : s_only ? -p.z : p.x - p.z, 1.0);
}
__global__
void f_ppm8_sobel_mask(float3* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{	
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	auto d = View2DSym<float3>(out, pitch_out, x, y, width, height);
	auto s = View2DSym<uchar3>(in, pitch_in, x, y, width, height);

	float Kx[9] = {
		+1, 0,-1,
		+2, 0,-2,
		+1, 0,-1,
	};
	float Ky[9] = {
		+1,+2,+1,
		 0, 0, 0,
		-1,-2,-1,
	};

	float3 Lx = make_float3(0,0,0);
	float3 Ly = make_float3(0,0,0);

	#pragma unroll
	for (int r=-1, i=0; r<2; r++, i++)
	{
		#pragma unroll
		for (int c=-1, j=0; c<2; c++, j++)
		{
			float  fx = Kx[i*3+j]/255.0;
			float  fy = Ky[i*3+j]/255.0;
			uchar3 ux = s(c,r);
			uchar3 uy = s(c,r);
			Lx.x += fx * ux.x;
			Lx.y += fx * ux.y;
			Lx.z += fx * ux.z;
			Ly.x += fy * uy.x;
			Ly.y += fy * uy.y;
			Ly.z += fy * uy.z;
		}
	}
#if 1
	Lx.x = Lx.y = Lx.z = (Lx.x+Lx.y+Lx.z)/3;
#endif
	float3 Lg = clamp(make_float3(
			pow(Lx.x*Lx.x + Ly.x*Ly.x, 0.5),
			pow(Lx.y*Lx.y + Ly.y*Ly.y, 0.5),
			pow(Lx.z*Lx.z + Ly.z*Ly.z, 0.5)
			), 0.0f, 1.0f);
	d(0,0) = Lg;
}

__global__
void f_ppm8_blend(
		uchar3* out, size_t pitch_out, 
		uchar3* a, size_t pitch_a, 
		uchar3* b, size_t pitch_b, 
		float3* mask, size_t pitch_mask, 
		size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	float3 f = *RC(float3, mask, pitch_mask, x, y, width, height);
	uchar3 va = *RC(uchar3, a, pitch_a, x, y, width, height);
	uchar3 vb = *RC(uchar3, b, pitch_b, x, y, width, height);
	float3 ia = make_float3(
			f.x * va.x / 255.0f,
			f.y * va.y / 255.0f,
			f.z * va.z / 255.0f);
	float3 ib = make_float3(
			(1-f.x) * vb.x / 255.0f,
			(1-f.y) * vb.y / 255.0f,
			(1-f.z) * vb.z / 255.0f);
	float3 blend = ia + ib;

	*RC(uchar3, out, pitch_out, x, y, width, height) = make_uchar3(blend.x*255, blend.y*255, blend.z*255);
}

__global__
void f_ppm8_bayer_pgm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	uchar3  p = *RC(uchar3, in, pitch_in, x, y, width, height);
	*RC(uint8_t, out, pitch_out, x, y, width, height) = IS_R(x,y)*p.x + IS_G(x,y)*p.y + IS_B(x,y)*p.z;
}

__global__
void f_pgm8_bayer_ppm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	uint8_t  p = *RC(uint8_t, in, pitch_in, x, y, width, height);
	*RC(uchar3, out, pitch_out, x, y, width, height) = make_uchar3(IS_R(x,y)*p, IS_G(x,y)*p, IS_B(x,y)*p);
}


__global__
void f_pgm8_debayer_bilinear_ppm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x >= width || y >= height) return;
	
	auto d = View2DSym<uchar3>(out, pitch_out, x, y, width, height);
	auto s = View2DSym<uint8_t>(in,  pitch_in,  x, y, width, height);

	d(0,0) = make_uchar3(
			(s( 0, 0 )), 
			(s(-1, 0 ) + s( 1, 0 ) + s( 0,-1 ) + s( 0, 1 )) >> 2, 
			(s(-1,-1 ) + s( 1,-1 ) + s(-1, 1 ) + s( 1, 1 )) >> 2);
		
	d(1, 1) = make_uchar3(
			(s( 0, 0 ) + s( 2, 0 ) + s( 0, 2 ) + s( 2, 2 )) >> 2, 
			(s( 0, 1 ) + s( 2, 1 ) + s( 1, 0 ) + s( 1, 2 )) >> 2, 
			(s( 1, 1 )));

	d(1, 0) = make_uchar3(
			(s( 0, 0 ) + s( 2, 0 )) >> 1, 
			(s( 1, 0 )), 
			(s( 1,-1 ) + s( 1, 1 )) >> 1);

	d(0, 1) = make_uchar3(
			(s( 0, 0 ) + s( 0, 2 )) >> 1, 
			(s( 0, 1 )), 
			(s(-1, 1 ) + s( 1, 1 )) >> 1);

}

__constant__ __device__ int32_t malvar[100];
void setupMalvar(cudaStream_t stream)
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
void f_pgm8_debayer_malvar_ppm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x >= width || y >= height) return;

	auto d = View2DSym<uchar3>(out, pitch_out, x, y, width, height);
	auto s = View2DSym<uint8_t>(in, pitch_in,  x, y, width, height);
	
	int3 trr = make_int3(s(0,0), 0, 0), trg = make_int3(0, s(1,0), 0),
	     tbg = make_int3(0, s(0,1), 0), tbb = make_int3(0, 0, s(1,1));

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

__global__
void f_pgm8_debayer_nn_ppm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x)*2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y)*2;
	if (x > width || y > height) return;

	auto d = View2DSym<uchar3>(out, pitch_out, x, y, width, height);
	auto s = View2DSym<uint8_t>(in, pitch_in,  x, y, width, height);

	d(0,0) = make_uchar3(s(0,0), (s(1,0) + s(0,1)) >> 1, s(1,1));
	d(1,0) = make_uchar3(s(2,0), (s(1,0) + s(2,1)) >> 1, s(1,1));
	d(0,1) = make_uchar3(s(0,2), (s(1,2) + s(0,1)) >> 1, s(1,1));
	d(1,1) = make_uchar3(s(2,2), (s(1,2) + s(2,1)) >> 1, s(1,1));
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
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
		    
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

		auto original = Image::load("kodak.ppm");
		original->copyToDevice(stream);
		original->printInfo();
	
		auto bayer = Image::create(Image::Type::pgm, original->width, original->height);
		f_ppm8_bayer_pgm8<<<gridSize, blockSize, 0, stream>>>(
				bayer->mem.device.data,
				bayer->mem.device.pitch,
				original->mem.device.data,
				original->mem.device.pitch,
				original->width,
				original->height
		);
		auto bayer_colored = Image::create(Image::Type::ppm, original->width, original->height);
		f_pgm8_bayer_ppm8<<<gridSize, blockSize, 0, stream>>>(
				bayer_colored->mem.device.data,
				bayer_colored->mem.device.pitch,
				bayer->mem.device.data,
				bayer->mem.device.pitch,
				bayer->width,
				bayer->height
		);
		
		setupMalvar(stream);
		auto debayer_lab = Image::create(Image::Type::lab, original->width, original->height);

		auto debayer0 = Image::create(Image::Type::ppm, original->width, original->height);
		auto debayer1 = Image::create(Image::Type::ppm, original->width, original->height);
		auto debayer2 = Image::create(Image::Type::ppm, original->width, original->height);
		auto debayer3 = Image::create(Image::Type::ppm, original->width, original->height);
		auto debayer4 = Image::create(Image::Type::ppm, original->width, original->height);
		auto debayer5 = Image::create(Image::Type::ppm, original->width, original->height);
		auto black    = Image::create(Image::Type::ppm, original->width, original->height);
		auto mask     = Image::create(Image::Type::raw, original->width, original->height, 3, 32);
		auto lab      = Image::create(Image::Type::lab, original->width, original->height);
		auto enhanced = Image::create(Image::Type::ppm, original->width, original->height);

		// NEAREST NEIGHBOR
		f_pgm8_debayer_nn_ppm8<<<gridSizeQ, blockSize, 0, stream>>>(
				debayer0->mem.device.data,
				debayer0->mem.device.pitch,
				bayer->mem.device.data,
				bayer->mem.device.pitch,
				bayer->width,
				bayer->height
		);
		debayer0->copyToHost(stream);
		
		// BILINEAR
		f_pgm8_debayer_bilinear_ppm8<<<gridSizeQ, blockSize, 0, stream>>>(
				debayer1->mem.device.data,
				debayer1->mem.device.pitch,
				bayer->mem.device.data,
				bayer->mem.device.pitch,
				bayer->width,
				bayer->height
		);
		debayer1->copyToHost(stream);
	
		// MALVAR
		f_pgm8_debayer_malvar_ppm8<<<gridSizeQ, blockSize, 0, stream>>>(
				debayer2->mem.device.data,
				debayer2->mem.device.pitch,
				bayer->mem.device.data,
				bayer->mem.device.pitch,
				bayer->width,
				bayer->height
		);
		debayer2->copyToHost(stream);
		
		// HAMILTON ADAMS
		HamiltonFilter hamilton;
		hamilton.source = bayer;
		hamilton.destination = debayer3;
		hamilton.run(stream);
		debayer3->copyToHost(stream);

		// GUNTURK
		GunturkFilter gunturk;
		gunturk.source = bayer;
		gunturk.destination = debayer4;
		gunturk.backend = HamiltonFilter();
		gunturk.run(stream);
		debayer4->copyToHost(stream);

		// My own tests
		PronkFilter pronk;
		pronk.source = bayer;
		pronk.destination = debayer5;
		pronk.run(stream);
		debayer5->copyToHost(stream);
		
		cudaDeviceSynchronize();
		printf("PSNR\n");
		printf("- Nearest:  %0.02f\n", debayer0->psnr(original));
		printf("- Bilinear: %0.02f\n", debayer1->psnr(original));
		printf("- Malvar:   %0.02f\n", debayer2->psnr(original));
		printf("- Adams:    %0.02f\n", debayer3->psnr(original));
		printf("- Gunturk:  %0.02f\n", debayer4->psnr(original));
		printf("- Pronk:    %0.02f\n", debayer5->psnr(original));
		printf("Creating screen\n");

		// SETUP DISPLAY
		CudaDisplay display(TITLE, WIDTH, HEIGHT); 
		display.cudaMap(stream);
		
		int i = 0;
		int count = 10;
		int scale = 1;
		int dx = 0, dy = 0;
		float ofs = 0;
		float angle = 0.00;
		float sat= 1, bri=1;
		float da=0, db=0;
		bool s_only = false, l_only = false;
		bool order = false;
		bool showEnhanced = false;

		Image* images[] = { original, bayer, bayer_colored,
			debayer0, debayer1, debayer2, debayer3, debayer4, debayer5, lab };
		while (true)
		{
			Image* img = images[i % count];
			if (img->type == Image::Type::ppm && showEnhanced)
			{
				img->toLab(lab, stream);

				f_cielab_enhance <<< gridSize, blockSize, 0, stream >>> (
					(float3*)lab->mem.device.data, lab->mem.device.pitch,
					lab->width, lab->height, angle, sat, bri, ofs, da, db);
			
				enhanced->fromLab(lab, stream);
				img = enhanced;
			}	
			
			switch (img->type)
			{
			case Image::Type::pgm:
				f_pgm8 <<< gridSize, blockSize, 0, stream >>> (
					display.CUDA.frame.data,
					display.CUDA.frame.pitch,
					img->mem.device.data,
					img->mem.device.pitch,
					img->width,
					img->height,
					scale,
					dx*scale, dy*scale);
				break;

			case Image::Type::ppm:
				f_ppm8 <<< gridSize, blockSize, 0, stream >>> (
					display.CUDA.frame.data,
					display.CUDA.frame.pitch,
					img->mem.device.data,
					img->mem.device.pitch,
					img->width,
					img->height,
					scale,
					dx*scale, dy*scale
				);
				break;

			case Image::Type::lab:
				f_cielab <<< gridSize, blockSize, 0, stream >>> (
					display.CUDA.frame.data,
					display.CUDA.frame.pitch,
					img->mem.device.data,
					img->mem.device.pitch,
					img->width,
					img->height,
					scale,
					dx*scale, dy*scale,
					l_only, s_only);
				break;

			}

			cudaStreamSynchronize(stream);
			display.cudaFinish(stream);
			display.render(stream);
		
			rc = cudaGetLastError();
			if (cudaSuccess != rc) throw "CUDA ERROR";

			if (int e = display.events()) 
			{
				if (e < 0)
				{
					display.cudaUnmap(stream);
					cudaStreamDestroy(stream);
					return 0;
				}
				else switch (e)
				{
					case ',': i--; if (i < 0) i=count-1; break;
					case '.': i++; if (i >= count) i=0; break;
					case '-': scale--; if (scale <= 0) scale = 1; break;
					case '=': scale++; if (scale >= 32) scale = 32; break;
					case 'w': dy-=10; break;
					case 's': dy+=10; break;
					case 'a': dx-=10; break;
					case 'd': dx+=10; break;
					case '[': bri *= 1.1f; break;
					case ']': bri /= 1.1f; break;
					case ';': sat *= 1.1f; break;
					case '\'':sat /= 1.1f; break;
					case 'n': ofs += 5;break;
					case 'm': ofs -= 5;break;
					case 'h': da += 3;break;
					case 'j': da -= 3;break;
					case 'y': db += 3;break;
					case 'u': db -= 5;break;
					case 'i': l_only = !l_only;break;
					case 'o': s_only = !s_only; break;
					case '0': 
					case '1':
					case '2':
					case '3':
					case '4':
					case '5':
					case '6':
					case '7':
					case '8':
					case '9': i = e - '0'; break;
					case 'q': 
						bri = 1, sat = 1,
						da = 0, db = 0, ofs = 0;
						break;
					case 'r':
						dx = 0, dy = 0, scale = 1;
					default: break;
				}
			}
			usleep(100000);
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
