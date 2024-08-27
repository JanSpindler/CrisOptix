#include <graph/CuEvent.h>
#include <util/custom_assert.h>

float CuEvent::GetElapsedTimeMs(const CuEvent& start, const CuEvent& stop)
{
	float elapsedTimeMs = 0.0f;
	ASSERT_CUDA(cudaEventElapsedTime(&elapsedTimeMs, start.m_Handle, stop.m_Handle));
	return elapsedTimeMs;
}

CuEvent::CuEvent()
{
	ASSERT_CUDA(cudaEventCreate(&m_Handle));
}

CuEvent::~CuEvent()
{
	ASSERT_CUDA(cudaEventDestroy(m_Handle));
}

void CuEvent::Sync() const
{
	ASSERT_CUDA(cudaEventSynchronize(m_Handle));
}

void CuEvent::Record(const cudaStream_t stream) const
{
	ASSERT_CUDA(cudaEventRecord(m_Handle, stream));
}
