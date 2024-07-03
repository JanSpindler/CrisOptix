#include <model/Texture.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <util/Log.h>
#include <cuda_runtime.h>
#include <util/custom_assert.h>

Texture::Texture(std::string filePath)
{
	// Rename file path if needed
	const size_t ddsStrPos = filePath.find(".dds");
	if (ddsStrPos != std::string::npos)
	{
		filePath.replace(ddsStrPos, 4, ".png");
	}

	// Load image into host memory
	int width = 0;
	int height = 0;
	int channelCount = 0;
	stbi_uc* data = stbi_load(filePath.c_str(), &width, &height, &channelCount, STBI_rgb_alpha);
	Log::Assert(data != nullptr, "STB failed to load image " + filePath);

	// Load onto device
	LoadToDevice(data, width, height);

	// Free host memory
	stbi_image_free(data);
}

Texture::~Texture()
{
	ASSERT_CUDA(cudaDestroyTextureObject(m_DeviceTex));
	ASSERT_CUDA(cudaFreeArray(m_DeviceData));
}

cudaTextureObject_t Texture::GetTextureObjext() const
{
	return m_DeviceTex;
}

void Texture::LoadToDevice(const stbi_uc* imageData, const int width, const int height)
{
	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
	ASSERT_CUDA(cudaMallocArray(&m_DeviceData, &channelDesc, width, height));

	const size_t pitch = width * 4 * sizeof(uint8_t);
	ASSERT_CUDA(cudaMemcpy2DToArray(m_DeviceData, 0, 0, imageData, pitch, pitch, height, cudaMemcpyHostToDevice));

	cudaResourceDesc resDesc{};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = m_DeviceData;

	cudaTextureDesc texDesc{};
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.addressMode[2] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.sRGB = 0;
	texDesc.borderColor[0] = 0.0f;
	texDesc.borderColor[1] = 0.0f;
	texDesc.borderColor[2] = 0.0f;
	texDesc.borderColor[3] = 0.0f;
	texDesc.normalizedCoords = 1;
	texDesc.maxAnisotropy = 1;
	texDesc.mipmapFilterMode = cudaFilterModePoint;
	texDesc.mipmapLevelBias = 0.0f;
	texDesc.minMipmapLevelClamp = 0;
	texDesc.maxMipmapLevelClamp = 99;
	texDesc.disableTrilinearOptimization = 0;
	texDesc.seamlessCubemap = 0;
	ASSERT_CUDA(cudaCreateTextureObject(&m_DeviceTex, &resDesc, &texDesc, nullptr));
}
