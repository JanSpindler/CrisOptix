#pragma once

#include <model/Material.h>
#include <model/Mesh.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <model/Texture.h>
#include <assimp/scene.h>
#include <glm/glm.hpp>

class Model
{
public:
	Model(const std::string& filePath, const bool flipUv, const SpecTexUsage specTexUsage, const OptixDeviceContext optixDeviceContext);
	~Model();

	void AddShader(Pipeline& pipeline, ShaderBindingTable& sbt) const;

	size_t GetMeshCount() const;
	const Mesh* GetMesh(const size_t idx) const;
	const std::vector<Mesh*>& GetMeshes() const;

	OptixTraversableHandle GetTraversHandle() const;

private:
	OptixTraversableHandle m_TraversHandle = 0;
	DeviceBuffer<uint8_t> m_AccelBuf{};
	mutable bool m_ShaderAdded = false;

	std::string m_FilePath{};
	std::string m_DirPath{};

	std::vector<Mesh*> m_Meshes{};
	std::vector<Material*> m_Materials{};
	std::unordered_map<std::string, Texture*> m_Textures{};

	void ProcessNode(aiNode* node, const aiScene* scene, const glm::mat4& parentT, const OptixDeviceContext optixDeviceContext);
	void LoadMesh(aiMesh* mesh, const aiScene* scene, const glm::mat4& t, const OptixDeviceContext optixDeviceContext);
	void LoadMaterials(const aiScene* scene, const SpecTexUsage specTexUsage);
	void BuildAccel(const OptixDeviceContext optixDeviceContext);
};
