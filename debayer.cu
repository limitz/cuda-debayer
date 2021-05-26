#include <debayer.h>

__global__
void f_debayer_pack(uint8_t* out, size_t pitch_out, uchar3* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x)*2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y)*2;
	if (x >= width || y >= height) return;
	auto d = View2DSym<uint8_t>(out, pitch_out, x, y, width, height);
	auto s = View2DSym<uchar3>(in, pitch_in, x, y, width, height);
	d(0,0) = s(0,0).x;
	d(1,0) = s(1,0).y;
	d(0,1) = s(0,1).y;
	d(1,1) = s(1,1).z;
}

__global__
void f_debayer_unpack(uchar3* out, size_t pitch_out, uint8_t* in, size_t pitch_in, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x)*2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y)*2;
	if (x >= width || y >= height) return;
	auto d = View2DSym<uchar3>(out, pitch_out, x, y, width, height);
	auto s = View2DSym<uint8_t>(in, pitch_in, x, y, width, height);
	d(0,0) = make_uchar3(s(0,0), 0, 0);
	d(1,0) = make_uchar3(0, s(1,0), 0);
	d(0,1) = make_uchar3(0, s(1,0), 0);
	d(1,1) = make_uchar3(0, 0, s(1,1));
}

__global__
void f_debayer_nn(uchar3* out, size_t pitch_out, uint8_t* in, size_t pitch_in, size_t width, size_t height)
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

__global__
void f_debayer_bilinear(uchar3* out, size_t pitch_out, uint8_t* in, size_t pitch_in, size_t width, size_t height)
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

Image* DebayerFilter::pack(Image* rgb, cudaStream_t stream)
{
	dim3 blockSize = {16,16};
	dim3 gridSize = {
		((int)rgb->width  + blockSize.x - 1) / blockSize.x,
		((int)rgb->height + blockSize.y - 1) / blockSize.y
	};

	dim3 gridSizeQ = {
		((int)rgb->width/2  + blockSize.x - 1) / blockSize.x,
		((int)rgb->height/2 + blockSize.y - 1) / blockSize.y
	};
	
	if (rgb->type != Image::Type::ppm || rgb->bpp != 8)
		throw "RGB image to pack must be 8bpp PPM";

	Image* destination = Image::create(Image::Type::pgm, rgb->width, rgb->height);
	
	f_debayer_pack <<< gridSize, blockSize, 0, stream >>> (
			(uint8_t*)destination->mem.device.data, destination->mem.device.pitch,
			(uchar3*)rgb->mem.device.data, rgb->mem.device.pitch,
			rgb->width, rgb->height);

	return destination;
}

Image* DebayerFilter::unpack(Image* b, cudaStream_t stream)
{
	dim3 blockSize = {16,16};
	dim3 gridSize = {
		((int)b->width  + blockSize.x - 1) / blockSize.x,
		((int)b->height + blockSize.y - 1) / blockSize.y
	};

	dim3 gridSizeQ = {
		((int)b->width/2  + blockSize.x - 1) / blockSize.x,
		((int)b->height/2 + blockSize.y - 1) / blockSize.y
	};
	if (b->type != Image::Type::pgm || b->bpp != 8)
		throw "Bayer image to unpack must be 8bpp PGM";
	
	Image* destination = Image::create(Image::Type::pgm, b->width, b->height);

	f_debayer_unpack <<< gridSize, blockSize, 0, stream >>> (
			(uchar3*)destination->mem.device.data, destination->mem.device.pitch,
			(uint8_t*)b->mem.device.data, b->mem.device.pitch,
			b->width, b->height);
	return destination;
}

void DebayerFilter::run(cudaStream_t stream)
{
	Filter::run(stream);

	if (source->type != Image::Type::pgm || source->bpp != 8)
		throw "Source image must be 8bpp PGM";

	if (destination->type != Image::Type::ppm || destination->bpp != 8)
		throw "Destination image must be 8bpp PPM";

	f_debayer_nn <<< gridSizeQ, blockSize, 0, stream >>> (
			(uchar3*)destination->mem.device.data, destination->mem.device.pitch,
			(uint8_t*)source->mem.device.data, source->mem.device.pitch,
			source->width, source->height);
}

void BilinearDebayerFilter::run(cudaStream_t stream)
{
	Filter::run(stream);

	if (source->type != Image::Type::pgm || source->bpp != 8)
		throw "Source image must be 8bpp PGM";

	if (destination->type != Image::Type::ppm || destination->bpp != 8)
		throw "Destination image must be 8bpp PPM";

	f_debayer_bilinear <<< gridSizeQ, blockSize, 0, stream >>> (
			(uchar3*)destination->mem.device.data, destination->mem.device.pitch,
			(uint8_t*)source->mem.device.data, source->mem.device.pitch,
			source->width, source->height);
}
