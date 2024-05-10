#include <model/Texture.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <util/Log.h>

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
	// TODO

	// Free host memory
	stbi_image_free(data);
}
