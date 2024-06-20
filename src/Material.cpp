#include <model/Material.h>

Material::Material(
	const glm::vec4& diffColor,
	const glm::vec4& specColor,
	const float roughness,
	const Texture* diffTex)
{
	MaterialSbtData data{};
	data.diffColor = glm::vec3(diffColor.x, diffColor.y, diffColor.z);
	data.specF0 = glm::vec3(specColor.x, specColor.y, specColor.z);
	data.roughTangent = roughness;
	data.roughBitangent = roughness;

	data.hasDiffTex = diffTex != nullptr;
	if (diffTex != nullptr) { data.diffTex = diffTex->GetTextureObjext(); }

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
