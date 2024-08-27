#pragma once

#include <cuda_runtime.h>

class CuEvent
{
public:
	static float GetElapsedTimeMs(const CuEvent& start, const CuEvent& stop);

	CuEvent();
	~CuEvent();

	void Sync() const;
	void Record(const cudaStream_t stream = 0) const;

private:
	cudaEvent_t m_Handle = nullptr;
};
