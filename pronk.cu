#include <pronk.h>
#include <view.h>

//__constant__ __device__ float sobel_kernel_x[9];

__device__
static inline float L(float3 rgb)
{
	return 100 * sqrt( 0.299*rgb.x*rgb.x + 0.587*rgb.y*rgb.y + 0.114*rgb.z*rgb.z );
}

__device__
static inline float A(float3 rgb)
{
	return 90 * (rgb.x - rgb.y);
}

__device__
static inline float B(float3 rgb)
{
	return 70 * ((rgb.x + rgb.y) - 2.0f * rgb.z);
}

__device__
static inline float3 Lab(float3 rgb)
{
	return make_float3(L(rgb), A(rgb), B(rgb));
}

__global__
void f_pronk(float3* dst, size_t dst_pitch, uint8_t* src, size_t src_pitch, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	auto d = View2DSym<float3> (dst, dst_pitch, x, y, width, height);
	auto s = View2DSym<uint8_t>(src, src_pitch, x, y, width, height);

	float v = (s(0,0) - 0.25f * (s(-2, 0) + s(2, 0) + s( 0,-2) + s(2, 0))) / 255.0f;
	//float3 rgb = clamp(make_float3(4*v, -4*v, 0), 0.0f, 1.0f);
	float lp = s(0,0)/255.0 - v;
	float3 rgb = clamp(make_float3(lp), 0.0f, 1.0f);
	d(0,0) = Lab(rgb);
}

__global__
void f_pronk_y(float3* dst, size_t dst_pitch, uint8_t* src, size_t src_pitch, size_t width, size_t height, float sharpness)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x)*2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y)*2;
	if (x >= width || y >= height) return;

	auto d = View2DSym<float3> (dst, dst_pitch, x, y, width, height);
	auto s = View2DSym<uint8_t>(src, src_pitch, x, y, width, height);

	float v1 = (s(0,0) - 0.25f * (s(-2, 0) + s(2, 0) + s( 0,-2) + s(0, 2)))*sharpness;
	float v2 = (s(1,1) - 0.25f * (s(-1, 1) + s(3, 1) + s( 1,-1) + s(1, 3)))*sharpness;
	float v3 = (s(1,0) - 0.25f * (s(-1, 0) + s(3, 0) + s( 1,-2) + s(1, 2)))*sharpness;
	float v4 = (s(0,1) - 0.25f * (s(-2, 1) + s(2, 1) + s( 0,-1) + s(0, 3)))*sharpness;
	
	float3 rgbr = clamp(make_float3(
		s(0,0),
		0.25f * (s(-1, 0) + s(1, 0) + s( 0,-1) + s(1, 0)),
		0.25f * (s(-1,-1) + s(1,-1) + s(-1, 1) + s(1, 1))),
		0.0f, 255.0f
	);
	float3 rgbb = clamp(make_float3(
		0.25f * (s(0,0) + s(2,0) + s(0,2) + s(2,2)),
		0.25f * (s(0,1) + s(2,1) + s(1,0) + s(1,2)),
		s(1,1)),
		0.0f, 255.0f
	);

	float3 rgbrg = clamp(make_float3(
		0.5f * (s(0, 0) + s(2, 0)),
		s(1,0) ,
		0.5f * (s(1,-1) + s(1, 1))),
		0.0f, 255.0f
	);
	float3 rgbbg = clamp(make_float3(
		0.5f * (s(0, 0) + s(0, 2)),
		s(0,1),
		0.5f * (s(-1,1) + s(1, 1))),
		0.0f, 255.0f
	);

	float3 Lr = Lab(v1+rgbr / 255.0);
	float3 Lb = Lab(v2+rgbb / 255.0);
	float3 Lrg = Lab(v3+rgbrg / 255.0);
	float3 Lbg = Lab(v4+rgbbg / 255.0);

	Lr  = make_float3(v1/5+Lr.x,  Lr.y,  Lr.z);
	Lb  = make_float3(v2/5+Lb.x,  Lb.y,  Lb.z);
	Lrg = make_float3(v3/5+Lrg.x, Lrg.y,  Lrg.z);
	Lbg = make_float3(v4/5+Lbg.x, Lbg.y,  Lbg.z);

	d(0,0) = Lr; 
	d(1,1) = Lb; 
	d(1,0) = Lrg; 
	d(0,1) = Lbg; 
}

void PronkFilter::run(cudaStream_t stream)
{
	Filter::run(stream);

	auto imm = Image::create(Image::Type::lab, source->width, source->height);

	if (source->width != destination->width || source->height != destination->height) 
		throw "Images not the same size";
	if (source->type != Image::Type::pgm || source->bpp != 8) 
		throw "Only 8 bit PGM input images are allowed";
	if (destination->type != Image::Type::ppm || destination->bpp != 8)
		throw "Only 8 bit PPM output images are allowed";

	dim3 gridSizeQ = {
		(((int)source->width /2 + blockSize.x - 1) / blockSize.x),
		(((int)source->height/2 + blockSize.y - 1) / blockSize.y)
	};
	dim3 gridSize = {
		(((int)source->width  + blockSize.x - 1) / blockSize.x),
		(((int)source->height + blockSize.y - 1) / blockSize.y)
	};

	f_pronk_y <<< gridSizeQ, blockSize, 0, stream >>> (
		(float3*)imm->mem.device.data, imm->mem.device.pitch,
		(uint8_t*)source->mem.device.data, source->mem.device.pitch,
		source->width, source->height, sharpness);

	destination->fromLab(imm, stream);
	delete imm;
}
