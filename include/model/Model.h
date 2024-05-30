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
	Model(const std::string& filePath, const bool flipUv, const OptixDeviceContext optixDeviceContext);
	~Model();

private:
	std::string m_FilePath{};
	std::string m_DirPath{};

	std::vector<Mesh*> m_Meshes{};
	std::vector<Material*> m_Materials{};
	std::unordered_map<std::string, Texture*> m_Textures{};
	
	OptixTraversableHandle m_TraversHandle = 0;
	DeviceBuffer<uint8_t> m_AccelStructBuf{};

	void ProcessNode(aiNode* node, const aiScene* scene, const glm::mat4& parentT);
	void LoadMesh(aiMesh* mesh, const aiScene* scene, const glm::mat4& t);
	void LoadMaterials(const aiScene* scene);
	void BuildAccelStructure(const OptixDeviceContext optixDeviceContext);
};
