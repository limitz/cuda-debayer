#include <gunturk.h>
#include <debayer.h>

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
__device__ size_t  gunturk_pitch;

void GunturkFilter::setup(cudaStream_t stream)
{
	if (!source) return;

	size_t width = source->width;
	size_t height = source->height;
	
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
void f_gunturk_gg1(uchar3* out, size_t pitch_out, uint8_t* in, size_t pitch_in, size_t width, size_t height)
{ 
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x >= width || y >= height) return;

	auto ca = View2DSym<float3>(gunturk_ca, gunturk_pitch, x, y, width, height);
	auto ch = View2DSym<float3>(gunturk_ch, gunturk_pitch, x, y, width, height);
	auto cv = View2DSym<float3>(gunturk_cv, gunturk_pitch, x, y, width, height);
	auto cd = View2DSym<float3>(gunturk_cd, gunturk_pitch, x, y, width, height);
	auto s  = View2DSym<uint8_t>(in, pitch_in, x, y, width, height);
	auto d  = View2DSym<uchar3>(out, pitch_out, x, y, width, height);
	
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
void f_gunturk_gg2(uchar3* out, size_t pitch_out, uint8_t* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	if (x >= width || y >= height) return;

	auto temp = View2DSym<float3>(gunturk_temp, gunturk_pitch, x, y, width, height);
	auto ca   = View2DSym<float3>(gunturk_ca, gunturk_pitch, x, y, width, height);
	auto ch   = View2DSym<float3>(gunturk_ch, gunturk_pitch, x, y, width, height);
	auto cv   = View2DSym<float3>(gunturk_cv, gunturk_pitch, x, y, width, height);
	auto cd   = View2DSym<float3>(gunturk_cd, gunturk_pitch, x, y, width, height);
	auto s    = View2DSym<uint8_t>(in, pitch_in, x, y, width, height);
	auto d    = View2DSym<uchar3>(out, pitch_out, x, y, width, height);

	float *g00 = gunturk_g00, *g10 = gunturk_g10, *g01 = gunturk_h01, *g11 = gunturk_g11;
	temp(0,0) =  temp(1,0) = temp(0,1) = temp(1,1) = make_float3(0,0,0);

	for (int r=-4; r<6; r+=2)
	{
		#pragma unroll
		for (int c=-4; c<6; c+=2, g00++, g10++, g01++, g11++)
		{
			temp(0,0).y += ca(c,r).y * *g00
			             + ch(c,r).y * *g10
			             + cv(c,r).y * *g01
			             + cd(c,r).y * *g11;
			
			temp(1,1).y += ca(c+1,r+1).y * *g00
			             + ch(c+1,r+1).y * *g10
			             + cv(c+1,r+1).y * *g01
			             + cd(c+1,r+1).y * *g11;
		}
	}
	d(0,0).x = s(0,0);
	d(1,1).z = s(1,1);
	d(0,0).y = clamp(temp(0,0).y, 0.f, 255.f);
	d(1,1).y = clamp(temp(1,1).y, 0.f, 255.f);
}

__global__
void f_gunturk_rb1(uchar3* out, size_t pitch_out, uint8_t* in, size_t pitch_in, size_t width, size_t height)
{ 
	int x = (blockIdx.x * blockDim.x + threadIdx.x) ;
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	auto ca = &View2DSym<float3>(gunturk_ca, gunturk_pitch, x, y, width, height)(0,0);
	auto ch = &View2DSym<float3>(gunturk_ch, gunturk_pitch, x, y, width, height)(0,0);
	auto cv = &View2DSym<float3>(gunturk_cv, gunturk_pitch, x, y, width, height)(0,0);
	auto cd = &View2DSym<float3>(gunturk_cd, gunturk_pitch, x, y, width, height)(0,0);
	auto d  = View2DSym<uchar3>(out, pitch_out, x, y, width, height);
	
	float *h00 = gunturk_h00, *h10 = gunturk_h10, *h01 = gunturk_h01, *h11 = gunturk_h11;

	*ca = *ch = *cv = *cd = make_float3(0,0,0); 
	
	for (int r=-1; r<2; r++)
	{
		#pragma unroll
		for (int c=-1; c<2; c++, h00++, h10++, h01++, h11++)
		{
			uchar3 v = d(c, r);
			float3 f = make_float3(v.x, v.y, v.z);
			*ca += f * *h00;
			*ch += f * *h10;
			*cv += f * *h01;
			*cd += f * *h11;
		}
	}
	ch->x = ch->z = ch->y;
	cv->x = cv->z = cv->y;
	cd->x = cd->z = cd->y;
}

__global__
void f_gunturk_rb2(void* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	auto ca = View2DSym<float3>(gunturk_ca, gunturk_pitch, x, y, width, height);
	auto ch = View2DSym<float3>(gunturk_ch, gunturk_pitch, x, y, width, height);
	auto cv = View2DSym<float3>(gunturk_cv, gunturk_pitch, x, y, width, height);
	auto cd = View2DSym<float3>(gunturk_cd, gunturk_pitch, x, y, width, height);
	auto t  = &View2DSym<float3>(gunturk_temp, gunturk_pitch, x, y, width, height)(0,0);
	auto d  = &View2DSym<uchar3>(out, pitch_out, x, y, width, height)(0,0);

	float *g00 = gunturk_g00, *g10 = gunturk_g10, *g01 = gunturk_h01, *g11 = gunturk_g11;
	
	*t = make_float3(0,0,0);
	
	for (int r=-2; r<3; r++)
	{
		#pragma unroll
		for (int c=-2; c<3; c++, g00++, g10++, g01++, g11++)
		{
			*t +=  ca(c,r) * *g00
			    + ch(c,r) * *g10
			    + cv(c,r) * *g01
			    + cd(c,r) * *g11;
		}
	}

	d->x = debayerIsRed(x,y) * d->x  + (1-debayerIsRed(x,y)) * clamp(t->x, 0.0f, 255.0f);
	d->z = debayerIsBlue(x,y) * d->z + (1-debayerIsBlue(x,y)) * clamp(t->z, 0.0f, 255.0f);
}

void GunturkFilter::run(cudaStream_t stream)
{
	Filter::run(stream);

	HamiltonFilter backend;
	backend.source = source;
	backend.destination = destination;
	backend.run(stream);
	
	
	f_gunturk_gg1 <<< gridSizeQ, blockSize, 0, stream >>> (
		(uchar3*) destination->mem.device.data,
		destination->mem.device.pitch,
		(uint8_t*) source->mem.device.data,
		source->mem.device.pitch,
		source->width,
		source->height
		);
		
	f_gunturk_gg2 <<< gridSizeQ, blockSize, 0, stream >>> (
		(uchar3*) destination->mem.device.data,
		destination->mem.device.pitch,
		(uint8_t*) source->mem.device.data,
		source->mem.device.pitch,
		source->width,
		source->height);

	for (int i=0; i<iterations; i++)
	{
		f_gunturk_rb1 <<< gridSize, blockSize, 0, stream>>> (	
			(uchar3*) destination->mem.device.data,
			destination->mem.device.pitch,
			(uint8_t*) source->mem.device.data,
			source->mem.device.pitch,
			source->width,
			source->height);
		
		f_gunturk_rb2 <<< gridSize, blockSize, 0, stream >>> (
			(uchar3*) destination->mem.device.data,
			destination->mem.device.pitch,
			(uint8_t*) source->mem.device.data,
			source->mem.device.pitch,
			source->width,
			source->height);
	}
	
}
