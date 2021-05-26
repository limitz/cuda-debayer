#pragma once
#include <filter.h>
#include <debayer.h>

class HamiltonFilter : public Filter
{
public:
	HamiltonFilter() : Filter("Hamilton-Adams") {}
	virtual void run(cudaStream_t stream) override;

	size_t threshold = 1;
};
