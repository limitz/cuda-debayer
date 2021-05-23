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

#define SYM(v, r) (min(max((int)(v), -(int)(v)), 2*(int)(r)-(int)(v)))
#define RC(type, var, pitch, x, y, width, height) ((type*)(((uint8_t*)(var)) + SYM((y),(height)) * (pitch)) + SYM((x),(width)))
#define IS_R(x,y) (~(x|y)&1)
#define IS_G(x,y) ((x^y)&1)
#define IS_B(x,y) (x&y&1)

template <typename T>
class View
{
public:
	__device__
	View(void* ptr, size_t ptr_pitch, int dx, int dy, size_t w, size_t h) 
	{
		data = (uint8_t*) ptr;
		pitch = ptr_pitch;
		width = w, height = h;
		x = dx, y = dy;
	}		
	
	__device__ ~View(){}

	__device__
	T& operator()(int dx, int dy)
	{
		return ((T*)(data + pitch * SYM(y + dy, height)))[SYM(x + dx, width)];
	}

	__device__
	View translated(int dx, int dy) const
	{
		return View(data, pitch, x+dx, y+dy, width, height);
	}

	__device__
	void translate(int dx, int dy)
	{
		y += dy;
		x += dx;
	}

	uint8_t* data;
	size_t pitch;
	size_t width;
	size_t height;
	int x;
	int y;
};

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
	
	auto d = View<uchar3>(out, pitch_out, x, y, width, height);
	auto s = View<uint8_t>(in,  pitch_in,  x, y, width, height);

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

	auto d = View<uchar3>(out, pitch_out, x, y, width, height);
	auto s = View<uint8_t>(in, pitch_in,  x, y, width, height);
	
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

	auto d = View<uchar3>(out, pitch_out, x, y, width, height);
	auto s = View<uint8_t>(in, pitch_in,  x, y, width, height);

	d(0,0) = make_uchar3(s(0,0), (s(1,0) + s(0,1)) >> 1, s(1,1));
	d(1,0) = make_uchar3(s(2,0), (s(1,0) + s(2,1)) >> 1, s(1,1));
	d(0,1) = make_uchar3(s(0,2), (s(1,2) + s(0,1)) >> 1, s(1,1));
	d(1,1) = make_uchar3(s(2,2), (s(1,2) + s(2,1)) >> 1, s(1,1));
}

__global__
void f_pgm8_debayer_adams_gg_ppm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x >= width || y >= height) return;
	
	auto d = View<uchar3>(out, pitch_out, x, y, width, height);
	auto s = View<uint8_t>(in, pitch_in,  x, y, width, height);
	
	// greens
	d(0,0) = make_uchar3(0, 0, 0);
	d(1,0) = make_uchar3(0, s(1,0), 0);
	d(0,1) = make_uchar3(0, s(0,1), 0);
	d(1,1) = make_uchar3(0, 0, 0);

	// greens at red / blue positions
	int treshold = 1;

	#pragma unroll
	for (int i=0; i<2; i++)
	{
		float green;
		int dh = abs(s(i-1,i)-s(i+1,i))+abs(2*s(i,i)-s(i+2,i)-s(i-2,i));
		int dv = abs(s(i,i-1)-s(i,i+1))+abs(2*s(i,i)-s(i,i+2)-s(i,i-2));
		
		if (dh > dv+treshold) 
			green = (s(i,i-1)+s(i,i+1))*0.5f+(2*s(i,i)-s(i,i-2)-s(i,i+2))*0.25f;
		else if (dv > dh+treshold) 
			green = (s(i-1,i)+s(i+1,i))*0.5f+(2*s(i,i)-s(i-2,i)-s(i+2,i))*0.25f;
		else
			green = (s(i,i-1)+s(i,i+1)+s(i-1,i)+s(i+1,i))*0.25f 
			      + (4*s(i,i)-s(i,i-2)-s(i,i+2)-s(i-2,i)-s(i+2,i))*0.125f;

		d(i,i).y = (uint8_t) clamp(green, 0.0f, 255.0f);
	}
}

__global__
void f_pgm8_debayer_adams_rb_ppm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x >= width || y >= height) return;
	
	auto d = View<uchar3>(out, pitch_out, x, y, width, height);
	auto s = View<uint8_t>(in, pitch_in,  x, y, width, height);
	
	d(0,0).x = (s(0,0));
	d(0,0).z = (s(-1,-1)+s(1,-1)+s(-1,1)+s(1,1)) >> 2;
	d(1,1).x = (s( 0, 0)+s(2, 0)+s( 0,2)+s(2,2)) >> 2;
	d(1,1).z = (s(1,1));
	d(1,0).x = (s( 0, 0)+s(2, 0)) >> 1;
	d(1,0).z = (s( 1,-1)+s(1, 1)) >> 1;
	d(0,1).x = (s( 0, 0)+s(0, 2)) >> 1;
	d(0,1).z = (s(-1, 1)+s(1, 1)) >> 1;
}

__constant__ __device__ float gunturk_h00[9];
__constant__ __device__ float gunturk_h10[9];
__constant__ __device__ float gunturk_h01[9];
__constant__ __device__ float gunturk_h11[9];
__constant__ __device__ float gunturk_g00[25];
__constant__ __device__ float gunturk_g10[25];
__constant__ __device__ float gunturk_g01[25];
__constant__ __device__ float gunturk_g11[25];
__device__ float3* gunturk_ca;
__device__ float3* gunturk_ch;
__device__ float3* gunturk_cv;
__device__ float3* gunturk_cd;
__device__ float3* gunturk_temp;
__device__ size_t gunturk_pitch;

void setupGunturk(cudaStream_t stream, size_t width, size_t height)
{
	float ph0[3] = {  0.25f,   0.5f,  0.25f };
	float ph1[3] = {  0.25f,  -0.5f,  0.25f };
	float pg0[5] = { -0.125f, 0.25f,  0.75f, 0.25f, -0.125f };
	float pg1[5] = {  0.125f, 0.25f, -0.75f, 0.25f,  0.125f };

	float ph00[9],  ph10[9],  ph01[9],  ph11[9];
	float pg00[25], pg10[25], pg01[25], pg11[25];

	for (int i=0; i<3; i++) for (int j=0; j<3; j++)
	{
		ph00[i*3+j] = ph0[i] * ph0[j];
		ph10[i*3+j] = ph1[i] * ph0[j];
		ph01[i*3+j] = ph0[i] * ph1[j];
		ph11[i*3+j] = ph1[i] * ph1[j];
	}
	for (int i=0; i<5; i++) for (int j=0; j<5; j++)
	{
		pg00[i*5+j] = pg0[i] * pg0[j];
		pg10[i*5+j] = pg1[i] * pg0[j];
		pg01[i*5+j] = pg0[i] * pg1[j];
		pg11[i*5+j] = pg1[i] * pg1[j];
	}

	int rc;
	float3 *ca, *ch, *cv, *cd, *temp;
	size_t pitch;
	
	rc  = cudaMallocPitch(&ca, &pitch, sizeof(float3) * width, height);
	rc |= cudaMallocPitch(&ch, &pitch, sizeof(float3) * width, height);
	rc |= cudaMallocPitch(&cv, &pitch, sizeof(float3) * width, height);
	rc |= cudaMallocPitch(&cd, &pitch, sizeof(float3) * width, height);
	rc |= cudaMallocPitch(&temp, &pitch, sizeof(float3) * width, height);
	if (cudaSuccess != rc) throw "Unable to allocate gunturk intermediate buffers";
	
	rc  = cudaMemcpyToSymbolAsync(gunturk_h00, &ph00, 9*sizeof(float), 0, cudaMemcpyHostToDevice, stream);
	rc |= cudaMemcpyToSymbolAsync(gunturk_h10, &ph10, 9*sizeof(float), 0, cudaMemcpyHostToDevice, stream);
	rc |= cudaMemcpyToSymbolAsync(gunturk_h01, &ph01, 9*sizeof(float), 0, cudaMemcpyHostToDevice, stream);
	rc |= cudaMemcpyToSymbolAsync(gunturk_h11, &ph11, 9*sizeof(float), 0, cudaMemcpyHostToDevice, stream);
	rc |= cudaMemcpyToSymbolAsync(gunturk_g00, &pg00, 25*sizeof(float), 0, cudaMemcpyHostToDevice, stream);
	rc |= cudaMemcpyToSymbolAsync(gunturk_g10, &pg10, 25*sizeof(float), 0, cudaMemcpyHostToDevice, stream);
	rc |= cudaMemcpyToSymbolAsync(gunturk_g01, &pg01, 25*sizeof(float), 0, cudaMemcpyHostToDevice, stream);
	rc |= cudaMemcpyToSymbolAsync(gunturk_g11, &pg11, 25*sizeof(float), 0, cudaMemcpyHostToDevice, stream);
	if (cudaSuccess != rc) throw "Unable to copy gunturk filters";
	
	rc  = cudaMemcpyToSymbolAsync(gunturk_ca, &ca, sizeof(float3*), 0, cudaMemcpyHostToDevice, stream);
	rc |= cudaMemcpyToSymbolAsync(gunturk_ch, &ch, sizeof(float3*), 0, cudaMemcpyHostToDevice, stream);
	rc |= cudaMemcpyToSymbolAsync(gunturk_cv, &cv, sizeof(float3*), 0, cudaMemcpyHostToDevice, stream);
	rc |= cudaMemcpyToSymbolAsync(gunturk_cd, &cd, sizeof(float3*), 0, cudaMemcpyHostToDevice, stream);
	rc |= cudaMemcpyToSymbolAsync(gunturk_temp, &temp, sizeof(float3*), 0, cudaMemcpyHostToDevice, stream);
	rc |= cudaMemcpyToSymbolAsync(gunturk_pitch, &pitch, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
	if (cudaSuccess != rc) throw "Unable to set gunturk intermediate buffers";
}

__global__
void f_pgm8_debayer_gunturk_gg1_ppm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{ 
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x >= width || y >= height) return;

	auto ca = View<float3>(gunturk_ca, gunturk_pitch, x, y, width, height);
	auto ch = View<float3>(gunturk_ch, gunturk_pitch, x, y, width, height);
	auto cv = View<float3>(gunturk_cv, gunturk_pitch, x, y, width, height);
	auto cd = View<float3>(gunturk_cd, gunturk_pitch, x, y, width, height);
	auto s = View<uint8_t>(in, pitch_in, x, y, width, height);
	auto d = View<uchar3>(out, pitch_out, x, y, width, height);
	
	float*h00 = gunturk_h00, *h10 = gunturk_h10, *h01 = gunturk_h01, *h11 = gunturk_h11;

	ca(0,0) = ch(0,0) = cv(0,0) = cd(0,0) = make_float3(0,0,0); 
	ca(0,1) = ch(0,1) = cv(0,1) = cd(0,1) = make_float3(0,0,0); 
	ca(1,0) = ch(1,0) = cv(1,0) = cd(1,0) = make_float3(0,0,0); 
	ca(1,1) = ch(1,1) = cv(1,1) = cd(1,1) = make_float3(0,0,0); 

	for (int r=-2; r<4; r+=2)
	{
		#pragma unroll
		for (int c=-2; c<4; c+=2, h00++, h10++, h01++, h11++)
		{
			uint8_t rr = s(c, r), bb = s(c+1, r+1);
			ca(0,0).x += rr * *h00,	ca(1,1).z += bb * *h00;
			ch(0,0).x += rr * *h10,	ch(1,1).z += bb * *h10;
			cv(0,0).x += rr * *h01, cv(1,1).z += bb * *h01;
			cd(0,0).x += rr * *h11,	cd(1,1).z += bb * *h11;
			ca(0,0).y += d(c, r).y * *h00;
			ca(1,1).y += d(c+1, r+1).y * *h00;
			
		}
	}
	ch(0,0).y = ch(0,0).x, cv(0,0).y = ch(0,0).x, cd(0,0).y = cd(0,0).x;
	ch(1,1).y = ch(1,1).z, cv(1,1).y = cv(1,1).z, cd(1,1).y = cd(1,1).z;
}


__global__
void f_pgm8_debayer_gunturk_gg2_ppm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x >= width || y >= height) return;

	auto ca = View<float3>(gunturk_ca, gunturk_pitch, x, y, width, height);
	auto ch = View<float3>(gunturk_ch, gunturk_pitch, x, y, width, height);
	auto cv = View<float3>(gunturk_cv, gunturk_pitch, x, y, width, height);
	auto cd = View<float3>(gunturk_cd, gunturk_pitch, x, y, width, height);
	auto temp = View<float3>(gunturk_temp, gunturk_pitch, x, y, width, height);
	auto s = View<uint8_t>(in, pitch_in, x, y, width, height);
	auto d = View<uchar3>(out, pitch_out, x, y, width, height);

	float *g00 = gunturk_g00, *g10 = gunturk_g10, *g01 = gunturk_h01, *g11 = gunturk_g11;
	
	temp(0,0) =  temp(1,0) = temp(0,1) = temp(1,1) = make_float3(0,0,0);

	for (int r=-4; r<6; r+=2)
	{
		#pragma unroll
		for (int c=-4; c<6; c+=2, g00++, g10++, g01++, g11++)
		{
			temp(0,0).y += ca(c,r).y * *g00;
			             + ch(c,r).y * *g10;
			             + cv(c,r).y * *g01;
			             + cd(c,r).y * *g11;
			
			temp(1,1).y += ca(c+1,r+1).y * *g00;
			             + ch(c+1,r+1).y * *g10;
			             + cv(c+1,r+1).y * *g01;
			             + cd(c+1,r+1).y * *g11;
		}
	}

	d(0,0).y = clamp(temp(0,0).y, 0.f, 255.f);
	d(1,1).y = clamp(temp(1,1).y, 0.f, 255.f);
}

__global__
void f_pgm8_debayer_gunturk_rb_ppm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x < 4 || y < 4 || x >= width-4 || y >= height-4) return;

	
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


		auto original = Image::load("castle.ppm");
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
		auto debayer0 = Image::create(Image::Type::ppm, original->width, original->height);
		auto debayer1 = Image::create(Image::Type::ppm, original->width, original->height);
		auto debayer2 = Image::create(Image::Type::ppm, original->width, original->height);
		auto debayer3 = Image::create(Image::Type::ppm, original->width, original->height);
		auto debayer4 = Image::create(Image::Type::ppm, original->width, original->height);

		f_pgm8_debayer_nn_ppm8<<<gridSizeQ, blockSize, 0, stream>>>(
				debayer0->mem.device.data,
				debayer0->mem.device.pitch,
				bayer->mem.device.data,
				bayer->mem.device.pitch,
				bayer->width,
				bayer->height
		);
		debayer0->copyToHost(stream);
		
		f_pgm8_debayer_bilinear_ppm8<<<gridSizeQ, blockSize, 0, stream>>>(
				debayer1->mem.device.data,
				debayer1->mem.device.pitch,
				bayer->mem.device.data,
				bayer->mem.device.pitch,
				bayer->width,
				bayer->height
		);
		debayer1->copyToHost(stream);
	
		
		f_pgm8_debayer_malvar_ppm8<<<gridSizeQ, blockSize, 0, stream>>>(
				debayer2->mem.device.data,
				debayer2->mem.device.pitch,
				bayer->mem.device.data,
				bayer->mem.device.pitch,
				bayer->width,
				bayer->height
		);
		debayer2->copyToHost(stream);
		
		f_pgm8_debayer_adams_gg_ppm8<<<gridSizeQ, blockSize, 0, stream>>>(
				debayer3->mem.device.data,
				debayer3->mem.device.pitch,
				bayer->mem.device.data,
				bayer->mem.device.pitch,
				bayer->width,
				bayer->height
		);
		f_pgm8_debayer_adams_rb_ppm8<<<gridSizeQ, blockSize, 0, stream>>>(
				debayer3->mem.device.data,
				debayer3->mem.device.pitch,
				bayer->mem.device.data,
				bayer->mem.device.pitch,
				bayer->width,
				bayer->height
		);
		debayer3->copyToHost(stream);

		setupGunturk(stream, bayer->width, bayer->height);
		f_pgm8_debayer_adams_gg_ppm8<<<gridSizeQ, blockSize, 0, stream>>>(
				debayer4->mem.device.data,
				debayer4->mem.device.pitch,
				bayer->mem.device.data,
				bayer->mem.device.pitch,
				bayer->width,
				bayer->height
		);
		f_pgm8_debayer_gunturk_gg1_ppm8<<<gridSizeQ, blockSize, 0, stream>>>(
				debayer4->mem.device.data,
				debayer4->mem.device.pitch,
				bayer->mem.device.data,
				bayer->mem.device.pitch,
				bayer->width,
				bayer->height
		);
		
		f_pgm8_debayer_gunturk_gg2_ppm8<<<gridSizeQ, blockSize, 0, stream>>>(
				debayer4->mem.device.data,
				debayer4->mem.device.pitch,
				bayer->mem.device.data,
				bayer->mem.device.pitch,
				bayer->width,
				bayer->height
		);
		debayer4->copyToHost(stream);

		CudaDisplay display(TITLE, WIDTH, HEIGHT); 
		cudaDeviceSynchronize();
		display.cudaMap(stream);
		
		printf("PSNR\n");
		printf("- Nearest:  %0.02f\n", debayer0->psnr(original));
		printf("- Bilinear: %0.02f\n", debayer1->psnr(original));
		printf("- Malvar:   %0.02f\n", debayer2->psnr(original));
		printf("- Adams:    %0.02f\n", debayer3->psnr(original));
		printf("- Gunturk:  %0.02f\n", debayer4->psnr(original));
		printf("Creating screen\n");


		int i = 0;
		int count = 7;
		int scale = 1;
		int dx = 0, dy = 0;

		Image* debayer[] = { bayer, bayer_colored, debayer0, debayer1, debayer2, debayer3, debayer4 };
		while (true)
		{
			if (!i)
			{
				f_pgm8<<<gridSize, blockSize, 0, stream>>>(
					display.CUDA.frame.data,
					display.CUDA.frame.pitch,
					bayer->mem.device.data,
					bayer->mem.device.pitch,
					bayer->width,
					bayer->height,
					scale,
					dx*scale, dy*scale
				);
			}
			else
			{
				f_ppm8<<<gridSize, blockSize, 0, stream>>>(
					display.CUDA.frame.data,
					display.CUDA.frame.pitch,
					debayer[i%count]->mem.device.data,
					debayer[i%count]->mem.device.pitch,
					debayer[i%count]->width,
					debayer[i%count]->height,
					scale,
					dx*scale, dy*scale
				);
			}

			cudaStreamSynchronize(stream);
			// Draw the pixelbuffer on screen
			display.cudaFinish(stream);
			display.render(stream);
		
			rc = cudaGetLastError();
			if (cudaSuccess != rc) throw "CUDA ERROR";

			// check escape pressed
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
					case 'w': dy+=10; break;
					case 's': dy-=10; break;
					case 'a': dx+=10; break;
					case 'd': dx-=10; break;
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
