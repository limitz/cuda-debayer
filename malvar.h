#pragma once
#include <filter.h>
#include <debayer.h>

class MalvarFilter : public Filter
{
public:
	MalvarFilter() : Filter("Marval") {}
	virtual void setup(cudaStream_t stream) override;
	virtual void run(cudaStream_t stream) override;
};
