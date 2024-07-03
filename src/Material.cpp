#include <model/Material.h>

Material::Material(
	const glm::vec4& diffColor,
	const glm::vec4& specColor,
	const glm::vec4& emissiveColor,
	const float roughness,
	const Texture* diffTex,
	const Texture* specTex,
	const Texture* roughTex)
{
	MaterialSbtData data{};
	data.diffColor = glm::vec3(diffColor.x, diffColor.y, diffColor.z);
	data.specF0 = glm::vec3(specColor.x, specColor.y, specColor.z);
	data.emissiveColor = glm::vec3(emissiveColor.x, emissiveColor.y, emissiveColor.z);
	data.roughness = roughness;
	
	data.hasDiffTex = diffTex != nullptr;
	if (diffTex != nullptr) { data.diffTex = diffTex->GetTextureObjext(); }

	data.hasSpecTex = specTex != nullptr;
	if (specTex != nullptr) { data.specTex = specTex->GetTextureObjext(); }

	data.hasRoughTex = roughTex != nullptr;
	if (roughTex != nullptr) { data.roughTex = roughTex->GetTextureObjext(); }

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
