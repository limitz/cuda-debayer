#pragma once
#include <filter.h>
#include <debayer.h>
#include <hamilton.h>

class GunturkFilter : public Filter
{
public:
	GunturkFilter() : Filter("Gunturk") {}
	virtual void setup(cudaStream_t stream) override;
	virtual void run(cudaStream_t stream) override;
	
	size_t iterations = 8;
};
