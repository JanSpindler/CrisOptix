#include <model/Model.h>
#include <util/Log.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <model/Vertex.h>

Model::Model(const std::string& filePath, const bool flipUv) :
	m_FilePath(filePath)
{
	// Assert
	Log::Assert(!filePath.empty());

	// Log
	Log::Info("Loading model " + m_FilePath);

	// Get directory path
	m_DirPath = m_FilePath.substr(0, m_FilePath.find_last_of('/'));

	// Import scene
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(m_FilePath, aiProcess_Triangulate | (flipUv ? aiProcess_FlipUVs : 0));
	if (scene == nullptr || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || scene->mRootNode == nullptr)
	{
		Log::Error("Assimp error: " + std::string(importer.GetErrorString()));
	}

	// Log scene info
	Log::Info("Model has " + std::to_string(scene->mNumMeshes) + " meshes");
	Log::Info("Model has " + std::to_string(scene->mNumMaterials) + " materials");

	// Process scene
	if (scene->HasMaterials()) { LoadMaterials(scene); }
	ProcessNode(scene->mRootNode, scene, glm::mat4(1.0f));
}

Model::~Model()
{
	for (Mesh* mesh : m_Meshes)
	{
		delete mesh;
	}

	for (Material* material : m_Materials)
	{
		delete material;
	}

	for (const std::pair<std::string, Texture*> textureEntry : m_Textures)
	{
		delete textureEntry.second;
	}
}

void Model::ProcessNode(aiNode* node, const aiScene* scene, const glm::mat4& parentT)
{
	// Assert
	Log::Assert(node != nullptr);
	Log::Assert(scene != nullptr);

	// Get transformation
	const aiMatrix4x4 localAiT = node->mTransformation;
	const glm::mat4 localT(
		localAiT.a1, localAiT.b1, localAiT.c1, localAiT.d1,
		localAiT.a2, localAiT.b2, localAiT.c2, localAiT.d2,
		localAiT.a3, localAiT.b3, localAiT.c3, localAiT.d3,
		localAiT.a4, localAiT.b4, localAiT.c4, localAiT.d4);
	const glm::mat4 totalT = parentT * localT;

	// Get meshes
	const size_t meshCount = node->mNumMeshes;
	//Log::Info("Node has " + std::to_string(meshCount) + " meshes");
	for (size_t meshIdx = 0; meshIdx < meshCount; ++meshIdx)
	{
		aiMesh* mesh = scene->mMeshes[node->mMeshes[meshIdx]];
		LoadMesh(mesh, scene, totalT);
	}

	// Get children
	const size_t childCount = node->mNumChildren;
	//Log::Info("\tNode has " + std::to_string(childCount) + " children");
	for (size_t childIdx = 0; childIdx < childCount; ++childIdx)
	{
		ProcessNode(node->mChildren[childIdx], scene, totalT);
	}
}

void Model::LoadMesh(aiMesh* mesh, const aiScene* scene, const glm::mat4& t)
{
	// Assert
	Log::Assert(mesh != nullptr);
	Log::Assert(scene != nullptr);

	// Normal matrix
	const glm::mat3 normalMat = glm::mat3(glm::transpose(glm::inverse(t)));

	//
	std::vector<Vertex> vertices{};
	std::vector<uint32_t> indices{};

	// For each vertex
	for (uint32_t vertIdx = 0; vertIdx < mesh->mNumVertices; ++vertIdx)
	{
		// Get vert pos
		glm::vec3 pos(0.0f);
		if (mesh->HasPositions())
		{
			pos.x = mesh->mVertices[vertIdx].x;
			pos.y = mesh->mVertices[vertIdx].y;
			pos.z = mesh->mVertices[vertIdx].z;
		}

		// Get vert normal
		glm::vec3 normal(0.0f);
		if (mesh->HasNormals())
		{
			normal.x = mesh->mNormals[vertIdx].x;
			normal.y = mesh->mNormals[vertIdx].y;
			normal.z = mesh->mNormals[vertIdx].z;
		}

		// Get vert uv
		glm::vec2 uv(0.0f);
		if (mesh->HasTextureCoords(0))
		{
			uv.x = mesh->mTextureCoords[0][vertIdx].x;
			uv.y = mesh->mTextureCoords[0][vertIdx].y;
		}

		// Post process
		pos = glm::vec3(t * glm::vec4(pos, 1.0f));
		normal = normalMat * normal;

		// Add vertex
		Vertex vert(pos, normal, uv);
		vertices.push_back(vert);
	}

	// Get triangle vertex indices
	for (uint32_t faceIdx = 0; faceIdx < mesh->mNumFaces; ++faceIdx)
	{
		aiFace& face = mesh->mFaces[faceIdx];
		for (uint32_t faceIndexIdx = 0; faceIndexIdx < face.mNumIndices; ++faceIndexIdx)
		{
			indices.push_back(face.mIndices[faceIndexIdx]);
		}
	}

	// Store mesh
	m_Meshes.push_back(new Mesh(vertices, indices, m_Materials[mesh->mMaterialIndex]));
}

void Model::LoadMaterials(const aiScene* scene)
{
	// Assert
	Log::Assert(scene != nullptr);
	Log::Assert(scene->HasMaterials());

	// Load material in correct order
	for (size_t matIdx = 0; matIdx < scene->mNumMaterials; ++matIdx)
	{
		// Get assimp material handle
		aiMaterial* material = scene->mMaterials[matIdx];
		Log::Info("Loading material " + std::string(material->GetName().C_Str()));

		// Diffuse color
		aiColor4D aiDiffuseColor{};
		glm::vec4 diffColor(1.0f);
		if (aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &aiDiffuseColor) == AI_SUCCESS)
		{
			diffColor = { aiDiffuseColor.r, aiDiffuseColor.g, aiDiffuseColor.b, aiDiffuseColor.a };
		}
		Log::Info("Diffuse Color (" +
			std::to_string(diffColor.r) + ", " +
			std::to_string(diffColor.g) + ", " +
			std::to_string(diffColor.b) + ", " +
			std::to_string(diffColor.a) + ")");

		// Diffuse texture
		const size_t diffTexCount = material->GetTextureCount(aiTextureType_DIFFUSE);
		Log::Info("Has " + std::to_string(diffTexCount) + " diffuse textures");
		Texture* diffTex = nullptr;
		if (diffTexCount > 0)
		{
			aiString texFileName{};
			material->GetTexture(aiTextureType_DIFFUSE, 0, &texFileName);
			Log::Info("Diffuse texture: " + std::string(texFileName.C_Str()));

			std::string texFilePath = m_DirPath + "/" + texFileName.C_Str();
			if (m_Textures.find(texFilePath) == m_Textures.end())
			{
				diffTex = new Texture(texFilePath);
				m_Textures.insert_or_assign(texFilePath, diffTex);
			}
			else
			{
				diffTex = m_Textures[texFilePath];
			}
		}

		// Add material
		m_Materials.push_back(new Material(diffColor, diffTex));
	}
}