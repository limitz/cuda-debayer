#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/stat.h>

#include <display.h>
#include <pthread.h>
#include <math.h>
#include <operators.h>
#include <image.h>
#include <sobel.h>
#include <hamilton.h>
#include <malvar.h>
#include <gunturk.h>
#include <pronk.h>
#include <view.h>
#include <gpu.h>

#ifndef TITLE
#define TITLE "CUDA DEBAYER DEMO"
#endif

__global__
void f_cielab_enhance(float3* lab, size_t pitch_in, size_t width, size_t height, float angle, float sat, float bri, float ofs, float da, float db)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	
	auto px = View2DSym<float3>(lab, pitch_in, x, y, width, height);

	px(0,0).y = cos(angle)  * px(0,0).y + sin(angle) * px(0,0).z;
	px(0,0).z = -sin(angle) * px(0,0).y + cos(angle) * px(0,0).z;
	px(0,0).x *= bri;
	px(0,0).x += ofs;
	px(0,0).y *= sat;
	px(0,0).z *= sat;
	px(0,0).y += da;
	px(0,0).z += db;
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
		rc = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		if (cudaSuccess != rc) throw "Unable to create CUDA stream";
		
		auto original = Image::load("kodak.ppm");
		original->copyToDevice(stream);
		original->printInfo();
		
		// Debayer source and destination images
		
		constexpr size_t debayer_count = 6;
		Image* debayer[debayer_count] = {0};
		for (int i=0; i<debayer_count; i++)
		{
			debayer[i] = Image::create(Image::Type::ppm, original->width, original->height);
		}

		// NEAREST NEIGHBOR
		DebayerFilter debayerNN;
		Image* bayer = DebayerFilter::pack(original, stream);
		Image* bayer_colored = DebayerFilter::unpack(bayer,stream);
		debayerNN.source = bayer;
		debayerNN.destination = debayer[0];
		debayerNN.run(stream);
		debayer[0]->copyToHost(stream);

		// BILINEAR
		BilinearDebayerFilter bilinear;
		bilinear.source = bayer;
		bilinear.destination = debayer[1];
		bilinear.run(stream);
		debayer[1]->copyToHost(stream);
	
		// MALVAR
		MalvarFilter malvar;
		malvar.source = bayer;
		malvar.destination = debayer[2];
		malvar.run(stream);
		debayer[2]->copyToHost(stream);
		
		// HAMILTON ADAMS
		HamiltonFilter hamilton;
		hamilton.source = bayer;
		hamilton.destination = debayer[3];
		hamilton.run(stream);
		debayer[3]->copyToHost(stream);

		// GUNTURK
		GunturkFilter gunturk;
		gunturk.source = bayer;
		gunturk.destination = debayer[4];
		gunturk.run(stream);
		debayer[4]->copyToHost(stream);

		// My own tests
		PronkFilter pronk;
		pronk.source = bayer;
		pronk.destination = debayer[5];
		pronk.run(stream);
		debayer[5]->copyToHost(stream);
	
		// Print statistics
		cudaDeviceSynchronize();
		printf("PSNR\n");
		for (size_t i=0; i<debayer_count; i++)
		{
			printf("- %d: %0.02f\n", i, debayer[i]->psnr(original));
		}

		// SETUP DISPLAY
		CudaDisplay disp(TITLE, original->width, original->height); 
		disp.cudaMap(stream);
		
		int i = 0;
		int count = 10;
		int scale = 1;
		int dx = 0, dy = 0;
		float ofs = 0;
		float angle = 0.00;
		float sat= 1, bri=1;
		float da=0, db=0;
		bool showEnhanced = true;


		auto black    = Image::create(Image::Type::ppm, original->width, original->height);
		auto mask     = Image::create(Image::Type::lab, original->width, original->height);
		auto lab1     = Image::create(Image::Type::lab, original->width, original->height);
		auto lab2     = Image::create(Image::Type::lab, original->width, original->height);
		auto enhanced = Image::create(Image::Type::ppm, original->width, original->height);
		
		Image* images[] = { original, bayer, bayer_colored,
			debayer[0], debayer[1], debayer[2], debayer[3], debayer[4], debayer[5], lab1 };

		while (true)
		{
			Image* img = images[i % count];
			if (img->type == Image::Type::ppm && showEnhanced)
			{
				img->toLab(lab1, stream);
				//black->toLab(lab2, stream);
				//conv(mask, lab, zipper, stream);
				//cmux(mask, mux,stream);
				//blend(lab, mask, lab2, lab, stream);
		
				dim3 blockSize = { 16, 16 };
				dim3 gridSize = { 
					((int)original->width  + blockSize.x - 1) / blockSize.x, 
					((int)original->height + blockSize.y - 1) / blockSize.y }; 

				
				f_cielab_enhance <<< gridSize, blockSize, 0, stream >>> (
					(float3*)lab1->mem.device.data, lab1->mem.device.pitch,
					lab1->width, lab1->height, angle, sat, bri, ofs, da, db);
				
				enhanced->fromLab(lab1, stream);
				img = enhanced;
			}	
			display(&disp, img, scale, dx, dy, 0, stream);

			cudaStreamSynchronize(stream);
			disp.cudaFinish(stream);
			disp.render(stream);
		
			rc = cudaGetLastError();
			if (cudaSuccess != rc) throw "CUDA ERROR";

			if (int e = disp.events()) 
			{
				if (e < 0)
				{
					disp.cudaUnmap(stream);
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
					case 'c': pronk.sharpness *= 1.1; pronk.run(stream); break;
					case 'v': pronk.sharpness /= 1.1; pronk.run(stream); break;

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
