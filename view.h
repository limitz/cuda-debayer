#pragma once
#include <cuda_runtime.h>
#include <operators.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define EdgeSym(v,r) (clamp((int)(v), -(int)(v), 2*(int)(r)-(int)(v)))
template <typename T>
__device__ __host__
inline T& ViewSingleSym(void* p, size_t ptr_pitch, int offsetx, int offsety, int maxx, int maxy)
{
	return ((T*)(((uint8_t*)p) + ptr_pitch * EdgeSym(offsety, maxy)))[EdgeSym(offsetx, maxx)];
}

template <typename T>
class View2DSym
{
public:
	void* ptr;
	size_t pitch;
	int max_x, max_y, offset_x, offset_y;
	
	__device__ __host__
	View2DSym(void* p, size_t ptr_pitch, int offsetx, int offsety, int maxx, int maxy)
	{
		ptr = p;
		pitch = ptr_pitch;
		max_x = maxx;
		max_y = maxy;
		offset_x = offsetx;
		offset_y = offsety;
	}

	__device__
	~View2DSym(){}

	__device__
	T& operator()(int x, int y)
	{
		return ((T*)(((uint8_t*)ptr) + pitch * EdgeSym(y + offset_y, max_y)))[EdgeSym(x + offset_x, max_x)];
	}

	__device__ __host__
	void translate(int dx, int dy)
	{
		offset_x += dx;
		offset_y += dy;
	}

};

