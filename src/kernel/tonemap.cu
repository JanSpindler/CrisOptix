#include <kernel/tonemap.h>
#include <cuda_runtime.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/compatibility.hpp>

template <typename T>
__host__ __device__ T CeilToMultiple(const T value, const T mod)
{
	return value - T(1) + mod - ((value - T(1)) % mod);
}

template <typename T>
__host__ __device__ T CeilDiv(const T num, const T den)
{
	return (num - T(1)) / den + T(1);
}

template <typename T>
__host__ __device__ T ApplySrgbGamma(const T& linear_color)
{
	// Proper sRGB curve...
	auto cond = glm::lessThan(linear_color, T(0.0031308f));
	auto if_true = 12.92f * linear_color;
	auto if_false = 1.055f * glm::pow(linear_color, T(1.0f / 2.4f)) - 0.055f;
	return glm::lerp(if_false, if_true, T(cond));
	// return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.0f/2.4f) - 0.055f;
}

__host__ __device__ glm::u8vec3 LinearToSrgb(const glm::vec3& linearColor)
{
	return static_cast<glm::u8vec3>(glm::clamp(ApplySrgbGamma(linearColor), 0.0f, 1.0f) * 255.0f);
}

__global__ void ToneMappingKernel(const CuBufferView<glm::vec3> inputHdr, CuBufferView<glm::u8vec3> outputLdr)
{
	const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= inputHdr.count)
	{
		return;
	}

	//outputLdr[idx] = LinearToSrgb(inputHdr[idx]);
	outputLdr[idx] = glm::u8vec3(100);
}

void ToneMapping(const CuBufferView<glm::vec3>& inputHdr, CuBufferView<glm::u8vec3>& outputLdr)
{
	const uint32_t blockSize = 512;
	const uint32_t blockCount = CeilDiv(inputHdr.count, blockSize);
	ToneMappingKernel<<<blockSize, blockCount>>>(inputHdr, outputLdr);
}
