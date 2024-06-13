#pragma once

#include <glm/glm.hpp>
#include <model/Texture.h>
#include <graph/brdf.h>
#include <graph/DeviceBuffer.h>

class Material
{
public:
	Material(const glm::vec4& diffColor, const Texture* diffTex);

	CUdeviceptr GetGgxDataPtr() const;

private:
	DeviceBuffer<GgxData> m_GgxDataBuf{};
};
