#include <model/Material.h>

Material::Material(const glm::vec4& diffColor, const Texture* diffTex)
{
	GgxData ggxData{};
	ggxData.diffColor = glm::vec3(diffColor.x, diffColor.y, diffColor.z);
	ggxData.specF0 = glm::vec3(0.3f, 0.0f, 0.0f);
	ggxData.roughTangent = 0.2f;
	ggxData.roughBitangent = 0.2f;
	m_GgxDataBuf.Alloc(1);
	m_GgxDataBuf.Upload(&ggxData);
}

CUdeviceptr Material::GetGgxDataPtr() const
{ 
	return m_GgxDataBuf.GetCuPtr();
}
