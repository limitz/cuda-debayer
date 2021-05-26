#pragma once

#include <cuda_runtime.h>
#include <operators.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <image.h>
#include <view.h>

class Filter
{
public:
	virtual void setup(cudaStream_t /*stream*/)
	{
	}

	virtual ~Filter() 
	{
		//if (_name) free(_name);
	}

	virtual void run(cudaStream_t stream) 
	{
		if (!source) 
			throw "Source not set";
	
		if (!destination) 
			throw "Destination not set";

		if (!_isInitialized)
		{
			setup(stream);
			_isInitialized = true;
		}
	
		gridSize = {
			((int)source->width  + blockSize.x - 1) / blockSize.x,
			((int)source->height + blockSize.y - 1) / blockSize.y
		};

		gridSizeQ = {
			((int)source->width/2  + blockSize.x - 1) / blockSize.x,
			((int)source->height/2 + blockSize.y - 1) / blockSize.y
		};
	}


	Image* source = nullptr;
	Image* destination = nullptr;

	dim3 blockSize = { 16, 16 };
	dim3 gridSize, gridSizeQ;

	Filter(const char* n = nullptr) 
	{
		if (n) _name = strdup(n);
	};

	const char* name() const { return _name; }

private:
	bool _isInitialized = false;
	char* _name;
};
