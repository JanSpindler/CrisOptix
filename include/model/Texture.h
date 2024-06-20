#pragma once

#include <string>
#include <stb_image.h>
#include <cuda_runtime.h>

class Texture
{
public:
	Texture(std::string filePath);
	~Texture();

	cudaTextureObject_t GetTextureObjext() const;

private:
	cudaArray_t m_DeviceData = nullptr;
	cudaTextureObject_t m_DeviceTex = 0;

	void LoadToDevice(const stbi_uc* imageData, const int width, const int height);
};
