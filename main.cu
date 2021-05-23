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
void f_pgm8(float4* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height, size_t scale)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	uint8_t px = ((uchar1*)(((uint8_t*) in) + pitch_in * y/scale))[x/scale].x / 255.0;
	out[y * pitch_out / sizeof(float4) + x] = make_float4(px, px, px, 1.0);
}

__global__
void f_ppm8(float4* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height, size_t scale)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	uchar3 p = ((uchar3*)(((uint8_t*) in) + pitch_in * (y/scale)))[x/scale];
	out[y * pitch_out / sizeof(float4) + x] = make_float4(p.x/255.0f, p.y/255.0f, p.z/255.0f, 1.0);
}

__global__
void f_ppm8_bayer_pgm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	uchar3  m = make_uchar3(~(x|y)&1, (x^y)&1, x&y&1); 
	uchar3  p = ((uchar3*)(((uint8_t*) in)  + pitch_in  * y))[x];
	uchar1* o = ((uchar1*)(((uint8_t*) out) + pitch_out * y)) + x;
	o[0] = make_uchar1(m.x*p.x + m.y*p.y + m.z*p.z);
}

__global__
void f_pgm8_debayer_bilinear_ppm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height, bool skipGreens = false)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x < 2 || y < 2 || x >= width-2 || y >= height-2) return;

	uchar3*  o1 = ((uchar3*)(((uint8_t*)out) + pitch_out * (y+0))) + x;	
	uchar3*  o2 = ((uchar3*)(((uint8_t*)out) + pitch_out * (y+1))) + x;	
	uint8_t* s1 = (((uint8_t*) in)  + pitch_in * y) + x;
	uint8_t* s0 = s1 - pitch_in;
	uint8_t* s2 = s1 + pitch_in;
	uint8_t* s3 = s2 + pitch_in;
	
	o1[0] = make_uchar3((s1[0]), (s1[-1]+s1[1]+s0[0]+s2[0])>>2, (s0[-1]+s0[1]+s2[-1]+s2[1])>>2);
	o2[1] = make_uchar3((s1[0]+s1[2]+s3[0]+s3[2])>>2, (s2[0]+s2[2]+s1[1]+s3[1])>>2, (s2[1]));
	o1[1] = make_uchar3((s1[0]+s1[2])>>1, (s1[1]), (s0[ 1]+s2[1])>>1);
	o2[0] = make_uchar3((s1[0]+s3[0])>>1, (s2[0]), (s2[-1]+s2[1])>>1);
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
	int rc = cudaMemcpyToSymbolAsync(malvar, &pmalvar, 100*sizeof(int32_t), 0, cudaMemcpyHostToDevice, stream);
	if (cudaSuccess != rc) throw "Unable to copy malvar kernels";
}

__global__
void f_pgm8_debayer_malvar_ppm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x < 2 || y < 2 || x >= width-2 || y >= height-2) return;

	uchar3*  o1 = ((uchar3*)(((uint8_t*)out) + pitch_out * (y+0))) + x;	
	uchar3*  o2 = ((uchar3*)(((uint8_t*)out) + pitch_out * (y+1))) + x;	
	uint8_t* s1 = ((uint8_t*) in) + pitch_in  * y + x;
	uint8_t* s2 = s1 + pitch_in;
	uint8_t* ss = s1 - 2 * pitch_in - 2;
	
	// red
	int3 trr = make_int3(s1[0], 0, 0);
	int3 trg = make_int3(0, s1[1], 0);
	int3 tbg = make_int3(0, s2[0], 0);
	int3 tbb = make_int3(0, 0, s2[1]);

	for (int i=0, *m=malvar; i<5; i++, ss += pitch_in)
	{
		uint8_t* t0 = ss;
		uint8_t* t1 = ss + pitch_in;

		#pragma unroll
		for (int j=0; j<5; j++, t0++, t1++, m++)
		{
			trr.y += m[ 0] * t0[0];
			trr.z += m[75] * t0[0];
			trg.x += m[25] * t0[1];
			trg.z += m[50] * t0[1];
			tbg.x += m[50] * t1[0];
			tbg.z += m[25] * t1[0];
			tbb.x += m[75] * t1[1];
			tbb.y += m[ 0] * t1[1];
		}
	}
	trr = clamp(trr, 0, 0xFF0);
	trg = clamp(trg, 0, 0xFF0);
	tbg = clamp(tbg, 0, 0xFF0);
	tbb = clamp(tbb, 0, 0xFF0);

	o1[0] = make_uchar3(trr.x, trr.y>>4, trr.z>>4);
	o1[1] = make_uchar3(trg.x>>4, trg.y, trg.z>>4);
	o2[0] = make_uchar3(tbg.x>>4, tbg.y, tbg.z>>4);
	o2[1] = make_uchar3(tbb.x>>4, tbb.y>>4, tbb.z);
}

__global__
void f_pgm8_debayer_nn_ppm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x)*2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y)*2;
	if (x < 2 || y < 2 || x >= width-2 || y >= height-2) return;

	uint8_t* s0 = ((uint8_t*) in) + pitch_in * (y+0) + x;
	uint8_t* s1 = ((uint8_t*) in) + pitch_in * (y+1) + x;
	uint8_t* s2 = ((uint8_t*) in) + pitch_in * (y+2) + x;
	uchar3*  o0 = ((uchar3*)(((uint8_t*)out) + pitch_out * (y+0))) + x;	
	uchar3*  o1 = ((uchar3*)(((uint8_t*)out) + pitch_out * (y+1))) + x;	

	o0[0] = make_uchar3(s0[0], (s0[1] + s1[0])/2, s1[1]);
	o0[1] = make_uchar3(s0[2], (s0[1] + s1[2])/2, s1[1]);
	o1[0] = make_uchar3(s2[0], (s2[1] + s1[0])/2, s1[1]);
	o1[1] = make_uchar3(s2[2], (s2[1] + s1[2])/2, s1[1]);
}

__global__
void f_pgm8_debayer_adams_gg_ppm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x < 2 || y < 2 || x >= width-2 || y >= height-2) return;
	
	uint8_t* s0 = ((uint8_t*) in) + pitch_in * y + x;
	uchar3*  o0 = ((uchar3*)(((uint8_t*)out) + pitch_out * (y+0))) + x;	
	uchar3*  o1 = ((uchar3*)(((uint8_t*)out) + pitch_out * (y+1))) + x;	
	
	// greens
	o0[1] = make_uchar3(0, s0[1], 0);
	o1[0] = make_uchar3(0, s0[pitch_in], 0);

	// greens at red / blue positions
	int treshold = 3;
	#pragma unroll
	for (int i=0; i<2; i++)
	{
		uint8_t* s = s0 + i * (pitch_in  + 1);
		uchar3*  o = i ? o1 + 1 : o0;
		int dh = abs(s[-1] - s[1])  + abs(2 * s[0] - s[2] - s[-2]);
		int dv = abs(s[-pitch_in] - s[pitch_in]) + abs(2 * s[0] - s[2*pitch_in] - s[-2*pitch_in]);
		float green;
		
		if (dh > dv+treshold) 
			green = (s[-pitch_in]+s[pitch_in]) * 0.5 
			      + (2 * s[0] - s[-2*pitch_in] - s[2*pitch_in]) * 0.25;

		else if (dv > dh+treshold) 
			green = (s[-1] + s[1]) * 0.5 
			      + (2 * s[0] - s[-2] - s[2]) * 0.25;

		else
			green = (s[-pitch_in] + s[pitch_in] + s[-1] + s[1]) * 0.25 
			      + (4 * s[0] - s[-2*pitch_in] - s[2*pitch_in] - s[-2] - s[2]) * 0.125;
		o[0].y = (uint8_t) clamp(green, 0.0f, 255.0f);
	}
}

__global__
void f_pgm8_debayer_adams_rb_ppm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x < 2 || y < 2 || x >= width-2 || y >= height-2) return;

	uchar3*  o1 = ((uchar3*)(((uint8_t*)out) + pitch_out * (y+0))) + x;	
	uchar3*  o2 = ((uchar3*)(((uint8_t*)out) + pitch_out * (y+1))) + x;	
	uint8_t* s1 = (((uint8_t*) in)  + pitch_in  * y) + x;
	uint8_t* s2 = s1 + pitch_in;
	uint8_t* s3 = s2 + pitch_in;
	uint8_t* s0 = s1 - pitch_in;
	
	o1[0] = make_uchar3((s1[0]), (o1[0].y), (s0[-1]+s0[1]+s2[-1]+s2[1])>>2);
	o2[1] = make_uchar3((s1[0]+s1[2]+s3[0]+s3[2])>>2, (o2[1].y), (s2[1]));
	o1[1] = make_uchar3((s1[0]+s1[2])>>1, (o1[1].y), (s0[ 1]+s2[1])>>1);
	o2[0] = make_uchar3((s1[0]+s3[0])>>1, (o2[0].y), (s2[-1]+s2[1])>>1);
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
	if (x < 4 || y < 4 || x >= width-4 || y >= height-4) return;

	float3* ca2 = ((float3*)(((uint8_t*)gunturk_ca) + gunturk_pitch * (y))) + x;
	float3* ch2 = ((float3*)(((uint8_t*)gunturk_ch) + gunturk_pitch * (y))) + x;
	float3* cv2 = ((float3*)(((uint8_t*)gunturk_cv) + gunturk_pitch * (y))) + x;
	float3* cd2 = ((float3*)(((uint8_t*)gunturk_cd) + gunturk_pitch * (y))) + x;
	float3* ca3 = ((float3*)(((uint8_t*)gunturk_ca) + gunturk_pitch * (y+1))) + x;
	float3* ch3 = ((float3*)(((uint8_t*)gunturk_ch) + gunturk_pitch * (y+1))) + x;
	float3* cv3 = ((float3*)(((uint8_t*)gunturk_cv) + gunturk_pitch * (y+1))) + x;
	float3* cd3 = ((float3*)(((uint8_t*)gunturk_cd) + gunturk_pitch * (y+1))) + x;

	uint8_t* s1 = (((uint8_t*) in)  + pitch_in  * y) + x;
	uint8_t* s2 = s1 + pitch_in;
	uint8_t* s0 = s1 - pitch_in;
	
	uchar3*  o1 = ((uchar3*)(((uint8_t*)out) + pitch_out * (y+0))) + x;	
	uchar3*  o2 = ((uchar3*)(((uint8_t*)out) + pitch_out * (y+1))) + x;	

	ca2[0] = ch2[0] = cv2[0] = cd2[0] = make_float3(0,0,0); 
	ca2[1] = ch2[1] = cv2[1] = cd2[1] = make_float3(0,0,0); 
	ca3[0] = ch3[0] = cv3[0] = cd3[0] = make_float3(0,0,0); 
	ca3[1] = ch3[1] = cv3[1] = cd3[1] = make_float3(0,0,0); 
	for (int i=0; i<3; i++)
	{
		#pragma unroll
		for (int j=0; j<3; j++)
		{
			uint8_t rr = s1[(2*i-2) * pitch_in + j*2-2];
			ca2[0].x += rr * gunturk_h00[i*3+j];
			ch2[0].x += rr * gunturk_h10[i*3+j];
			cv2[0].x += rr * gunturk_h01[i*3+j];
			cd2[0].x += rr * gunturk_h11[i*3+j];
			
			uint8_t rg = ((uchar3*)(((uint8_t*)o1) + (2*i-2) * pitch_out))[j*2-2].y;
			ca2[0].y += rg * gunturk_h00[i*3+j];

			uint8_t bb = s1[(2*i-1) * pitch_in + j*2-1];
			ca3[1].z += bb * gunturk_h00[i*3+j];
			ch3[1].z += bb * gunturk_h10[i*3+j];
			cv3[1].z += bb * gunturk_h01[i*3+j];
			cd3[1].z += bb * gunturk_h11[i*3+j];
			
			uint8_t bg = ((uchar3*)(((uint8_t*)o1) + (2*i-1) * pitch_out))[j*2-1].y;
			ca3[1].y += bg * gunturk_h00[i*3+j];
		}
	}

	ch2[0].y = ch2[0].x;
	cv2[0].y = ch2[0].x;
	cd2[0].y = cd2[0].x;
	ch3[1].y = ch3[1].z;
	cv3[1].y = cv3[1].z;
	cd3[1].y = cd3[1].z;

}


__global__
void f_pgm8_debayer_gunturk_gg2_ppm8(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x < 4 || y < 4 || x >= width-4 || y >= height-4) return;

	float3* ca = ((float3*)(((uint8_t*)gunturk_ca) + gunturk_pitch * (y-4))) + x;
	float3* ch = ((float3*)(((uint8_t*)gunturk_ch) + gunturk_pitch * (y-4))) + x;
	float3* cv = ((float3*)(((uint8_t*)gunturk_cv) + gunturk_pitch * (y-4))) + x;
	float3* cd = ((float3*)(((uint8_t*)gunturk_cd) + gunturk_pitch * (y-4))) + x;
	float3* temp0 = ((float3*)(((uint8_t*)gunturk_temp) + gunturk_pitch * y)) + x;
	float3* temp1 = ((float3*)(((uint8_t*)gunturk_temp) + gunturk_pitch * (y+1))) + x;
	
	temp0[0] =  temp0[1] = temp1[0] = temp1[1] = make_float3(0,0,0);
	for (int i=0; i<5; i++)
	{
		float3* ca0 = ((float3*)(((uint8_t*)ca) + 2*i * gunturk_pitch));
		float3* ch0 = ((float3*)(((uint8_t*)ch) + 2*i * gunturk_pitch));
		float3* cv0 = ((float3*)(((uint8_t*)cv) + 2*i * gunturk_pitch));
		float3* cd0 = ((float3*)(((uint8_t*)cd) + 2*i * gunturk_pitch));

		float3* ca1 = ((float3*)(((uint8_t*)ca) + (2*i+1) * gunturk_pitch));
		float3* ch1 = ((float3*)(((uint8_t*)ch) + (2*i+1) * gunturk_pitch));
		float3* cv1 = ((float3*)(((uint8_t*)cv) + (2*i+1) * gunturk_pitch));
		float3* cd1 = ((float3*)(((uint8_t*)cd) + (2*i+1) * gunturk_pitch));
		#pragma unroll
		for (int j=0; j<5; j++)
		{
			temp0[0].y += ca0[j*2-4].y * gunturk_g00[i*5+j];
			temp0[0].y += ch0[j*2-4].y * gunturk_g10[i*5+j];
			temp0[0].y += cv0[j*2-4].y * gunturk_g01[i*5+j];
			temp0[0].y += cd0[j*2-4].y * gunturk_g11[i*5+j];
			temp1[1].y += ca1[j*2-3].y * gunturk_g00[i*5+j];
			temp1[1].y += ch1[j*2-3].y * gunturk_g10[i*5+j];
			temp1[1].y += cv1[j*2-3].y * gunturk_g01[i*5+j];
			temp1[1].y += cd1[j*2-3].y * gunturk_g11[i*5+j];
		}
	}

	uchar3*  o1 = ((uchar3*)(((uint8_t*)out) + pitch_out * (y+0))) + x;	
	uchar3*  o2 = ((uchar3*)(((uint8_t*)out) + pitch_out * (y+1))) + x;	
	o1[0] = make_uchar3(0, clamp(temp0[0].y, 0.f, 255.f),0);
	o2[1] = make_uchar3(0, clamp(temp1[1].y, 0.f, 255.f),0);
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
		Image* debayer[] = { debayer4, debayer0, debayer1, debayer2, debayer3 };
		while (true)
		{
#if 0
			f_pgm8<<<gridSize, blockSize, 0, stream>>>(
				display.CUDA.frame.data,
				display.CUDA.frame.pitch,
				bayer->mem.device.data,
				bayer->mem.device.pitch,
				bayer->width,
				bayer->height
			);
#else
			f_ppm8<<<gridSize, blockSize, 0, stream>>>(
				display.CUDA.frame.data,
				display.CUDA.frame.pitch,
				debayer[i%4]->mem.device.data,
				debayer[i%4]->mem.device.pitch,
				debayer[i%4]->width,
				debayer[i%4]->height,
				1
			);
#endif

			cudaStreamSynchronize(stream);
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
			usleep(1000000);
			i++;
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
