#include <filter.h>

class SobelFilter
{
public:
	SobelFilter();
	~SobelFilter();

	Image* source;
	Image* destination;
	dim3   blockSize;
	bool   avgChannels;
	float  power;
	void run(cudaStream_t stream);
};
