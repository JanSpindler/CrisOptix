#include <model/Material.h>

Material::Material(const glm::vec4& diffColor, const Texture* diffTex)
{
	MaterialSbtData data{};
	data.diffColor = glm::vec3(diffColor.x, diffColor.y, diffColor.z);
	data.specF0 = glm::vec3(0.3f, 0.0f, 0.0f);
	data.roughTangent = 0.2f;
	data.roughBitangent = 0.2f;
	m_SbtDataBuf.Alloc(1);
	m_SbtDataBuf.Upload(&data);
}

void Material::AddShader(Pipeline& pipeline, ShaderBindingTable& sbt)
{
	const OptixProgramGroup pg = pipeline.AddCallableShader({ "brdf.ptx", "__direct_callable__ggx" });
	m_EvalSbtIdx = sbt.AddCallableEntry(pg, ToVecByte(m_SbtDataBuf.GetCuPtr()));
}

uint32_t Material::GetEvalSbtIdx() const
{
	return m_EvalSbtIdx;
}
