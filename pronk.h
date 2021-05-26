#pragma once
#include <filter.h>
#include <debayer.h>

class PronkFilter : public Filter
{
public:
	PronkFilter() : Filter("Pronk test") {}
	virtual void run(cudaStream_t stream) override;

	float sharpness = 0.001;
};
