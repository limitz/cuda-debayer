#include <image.h>

static const char* strrstr(const char* c, const char* find)
{
	if (!c || !find) return nullptr;
	if (strlen(find) == 0) return c + strlen(c);
	if (strlen(c) < strlen(find)) return nullptr;

	for (int i=strlen(c)-strlen(find); i >= 0; i--)
	{
		if (!memcmp(c + i, find, strlen(find))) return c + i;
	}
	return nullptr;
}

Image::~Image()
{
	if (_filename) free(_filename);
	if (mem.host.data) cudaFreeHost(mem.host.data);
	if (mem.device.data) cudaFree(mem.device.data);
}

Image::Image()
{
	_filename = nullptr;
	mem.host.data = nullptr;
	mem.device.data = nullptr;
	filename = nullptr;
	width = 0;
	height = 0;
	mem.host.pitch = 0;
	mem.device.pitch = 0;
	channels = 0;
	bpp = 0;
	type = Type::unknown;
}

void Image::copyToHost(cudaStream_t stream)
{
	int rc = cudaMemcpy2DAsync(
			mem.host.data, mem.host.pitch, 
			mem.device.data, mem.device.pitch, 
			width * (bpp >> 3) * channels, 
			height,
			cudaMemcpyDeviceToHost, 
			stream); 
	if (cudaSuccess != rc) throw "Unable to copy from device to host";
}
void Image::copyToDevice(cudaStream_t stream)
{
	int rc = cudaMemcpy2DAsync(
			mem.device.data, mem.device.pitch, 
			mem.host.data, mem.host.pitch, 
			width * (bpp >> 3) * channels, 
			height,
			cudaMemcpyHostToDevice, 
			stream); 
	if (cudaSuccess != rc) throw "Unable to copy from host to device";
}

void Image::printInfo()
{
	printf("IMAGE %s\n", filename);
	const char* typeName;
	switch (type)
	{
		case Type::unknown: typeName = "unknown"; break;
		case Type::jpeg: typeName = "JPEG"; break;
		case Type::ppm:  typeName = "PPM"; break;
		case Type::pgm:  typeName = "PGM"; break;
		default: typeName = "INVALID!"; break;
	}
	printf("- TYPE:  %s\n", typeName);
	printf("- SIZE:  %lu x %lu\n", width, height);
	printf("- PITCH: %lu (dev), %lu (host)\n", mem.device.pitch, mem.host.pitch);
	printf("- RANGE: 0x%lX\n", range);
	printf("\n");
	fflush(stdout);
}

Image* Image::create(Type type, size_t width, size_t height, size_t bpp)
{
	int rc;
	auto result = new Image();
	

	result->width = width;
	result->height = height;
	result->type = type;
	result->bpp = bpp;

	switch (type)
	{
		case Type::jpeg: 
		case Type::ppm:
			result->channels=3; 
			break;
		case Type::pgm:
			result->channels=1;
			break;
		default: throw "Invalid image type";
	}
	result->range = (1 << result->bpp) - 1;
	result->mem.host.pitch = width * (bpp >> 3) * result->channels;
	rc = cudaMallocHost(&result->mem.host.data, result->mem.host.pitch * height);
	if (cudaSuccess != rc) throw "Unable to allocate host memory for image";

	rc = cudaMallocPitch(&result->mem.device.data, &result->mem.device.pitch, 
			width * (bpp >> 3) * result->channels, height);
	if (cudaSuccess != rc) throw "Unable to allocate device memory for image";
	
	return result;
}

void Image::loadPPM()
{
	int rc;

	FILE* f = fopen(filename, "rb");
	if (!f) throw "Unable to open file";

	rc = fscanf(f, "P6 %lu %lu %lu \n", &width, &height, &range);
	if (rc <= 0) throw "Unable to read PPM header";
	
	type = Type::ppm;
	channels = 3;
	if (range < 256) bpp = 8;
	else bpp = 16;

	mem.host.pitch = width * (bpp >> 3) * channels;
	rc = cudaMallocHost(&mem.host.data, mem.host.pitch * height);
	if (cudaSuccess != rc) throw "Unable to allocate host memory for image";

	rc = cudaMallocPitch(&mem.device.data, &mem.device.pitch, width * (bpp >> 3) * channels, height);
	if (cudaSuccess != rc) throw "Unable to allocate device memory for image";

	rc = fread(mem.host.data, 1, mem.host.pitch * height, f);
	if (rc <= 0) throw "Unable to read image data from PPM";

	fclose(f);
}

void Image::loadPGM()
{
}

void Image::loadJPG()
{
}

Image* Image::load(const char* filename)
{
	auto result = new Image();
	result->filename = result->_filename = strdup(filename);

	const char* extension = strrstr(filename, ".");

	if (!strcmp(extension, ".jpeg")) result->loadJPG();
	if (!strcmp(extension, ".jpg"))  result->loadJPG();
	if (!strcmp(extension, ".ppm"))  result->loadPPM();
	if (!strcmp(extension, ".pgm"))  result->loadPGM();

	return result;
}

JpegCodec::JpegCodec()
{
	_width = 0;
	_height = 0;
	_channels = 0;
	_buffer = nullptr;
	_scanlines = nullptr;
	
	_dinfo.err = jpeg_std_error(&_djerr);
	_cinfo.err = jpeg_std_error(&_cjerr);
}
	
JpegCodec::~JpegCodec()
{
	free(_buffer);
	free(_scanlines);
}

void JpegCodec::prepare(int width, int height, int channels, int quality)
{
	if (channels != 3) throw "Not implemented channels != 3";

	_width = width;
	_height = height;
	_channels = channels;

	_buffer = (uint8_t*) malloc(_width * _height * _channels);
	if (!_buffer) throw "Unable to allocate intermediate buffer";

	_scanlines = (JSAMPARRAY) malloc( sizeof(JSAMPROW) * height);
	if (!_scanlines)
	{
		free(_buffer);
		throw "Unable to allocate scanlines structure";
	}

	for (size_t i=0; i<_height; i++)
	{
		_scanlines[i] = (JSAMPROW) (_buffer + i * _width * _channels);
	}

	jpeg_create_decompress(&_dinfo);
	jpeg_create_compress(&_cinfo);
	
	_cinfo.image_width = _width;
	_cinfo.image_height = height;
	_cinfo.input_components = 3;
	_cinfo.in_color_space = JCS_RGB; 
	jpeg_set_defaults(&_cinfo);
	jpeg_set_quality(&_cinfo, quality, 1);
}

void JpegCodec::unprepare()
{
	jpeg_destroy_decompress(&_dinfo);
	jpeg_destroy_compress(&_cinfo);
}

void JpegCodec::encodeCPU(void* dst, size_t *size)
{
	//cudaMemcpyAsync(_buffer, src, _width * _height * _channels, cudaMemcpyDeviceToHost, stream);
	//cudaStreamSynchronize(stream);
	
	jpeg_mem_dest(&_cinfo, (uint8_t**)&dst, size);
	jpeg_start_compress(&_cinfo, 1);
	while (_cinfo.next_scanline < _cinfo.image_height)
	{
		jpeg_write_scanlines(&_cinfo, _scanlines + _cinfo.next_scanline, _cinfo.image_height - _cinfo.next_scanline);
	}
	jpeg_finish_compress(&_cinfo);
}

void JpegCodec::decodeToDeviceMemoryCPU(void* dst, const void* src, size_t size, cudaStream_t stream)
{
	jpeg_mem_src(&_dinfo, (uint8_t*)src, size);
	jpeg_read_header(&_dinfo, 1);
	jpeg_calc_output_dimensions(&_dinfo);

	if (_dinfo.output_width != _width 
	||  _dinfo.output_height != _height
	||  _dinfo.output_components != (int) _channels)
	{
		jpeg_abort_decompress(&_dinfo);
		throw "Invalid image format";
	}
	jpeg_start_decompress(&_dinfo);
	while (_dinfo.output_scanline < _dinfo.output_height)
	{
		jpeg_read_scanlines(&_dinfo, _scanlines + _dinfo.output_scanline,_dinfo.output_height - _dinfo.output_scanline);
	}
	jpeg_finish_decompress(&_dinfo);

	cudaMemcpyAsync(dst, _buffer, _width * _height * _channels, cudaMemcpyHostToDevice, stream);
}

#if USE_NVJPEG
void JpegCodec::decodeToDeviceMemoryGPU(void* dst, const void* src, size_t size, cudaStream_t stream)
{
	int rc;
	
	nvjpegHandle_t handle;
	rc = nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, NULL, NULL, 0, &handle);
	if (cudaSuccess != rc) throw "Unable to create nvjpeg handle";

	int channels;
	int widths[NVJPEG_MAX_COMPONENT];
	int heights[NVJPEG_MAX_COMPONENT];
	nvjpegChromaSubsampling_t subsampling;
	nvjpegJpegState_t state;
	nvjpegOutputFormat_t fmt = NVJPEG_OUTPUT_RGBI;
	nvjpegJpegStateCreate(handle, &state);
	nvjpegGetImageInfo(handle, (uint8_t*) src, size, &channels, &subsampling, widths, heights);

	if (widths[0] != (int)_width
	||  heights[0] != (int)_height)
	{
		nvjpegJpegStateDestroy(state);
		nvjpegDestroy(handle);
		throw "Invalid image format";
	}

	nvjpegImage_t output;
	output.channel[0] = (uint8_t*) dst;
	output.pitch[0] = widths[0] * _channels;

	nvjpegDecode(handle, state, (uint8_t*)src, size, fmt, &output, stream);
	nvjpegJpegStateDestroy(state);
	nvjpegDestroy(handle);

}
#endif
