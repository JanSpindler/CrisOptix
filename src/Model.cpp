#include <model/Model.h>
#include <util/Log.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <model/Vertex.h>
#include <optix_stubs.h>
#include <optix_host.h>

Model::Model(const std::string& filePath, const bool flipUv, const SpecTexUsage specTexUsage, const OptixDeviceContext optixDeviceContext) :
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
	const aiScene* scene = importer.ReadFile(m_FilePath, 
		aiProcess_Triangulate 
		| aiProcess_GenNormals
		| aiProcess_ValidateDataStructure
		| aiProcess_CalcTangentSpace
		| (flipUv ? aiProcess_FlipUVs : 0));
	if (scene == nullptr || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || scene->mRootNode == nullptr)
	{
		Log::Error("Assimp error: " + std::string(importer.GetErrorString()));
	}

	// Log scene info
	Log::Info("Model has " + std::to_string(scene->mNumMeshes) + " meshes");
	Log::Info("Model has " + std::to_string(scene->mNumMaterials) + " materials");

	// Process scene
	if (scene->HasMaterials()) { LoadMaterials(scene, specTexUsage); }
	ProcessNode(scene->mRootNode, scene, glm::mat4(1.0f), optixDeviceContext);

	// Build accel
	BuildAccel(optixDeviceContext);
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

void Model::AddShader(Pipeline& pipeline, ShaderBindingTable& sbt) const
{
	if (m_ShaderAdded) { return; }
	m_ShaderAdded = true;

	// Call material before mesh
	for (Material* material : m_Materials)
	{
		material->AddShader(pipeline, sbt);
	}

	for (Mesh* mesh : m_Meshes)
	{
		mesh->AddShader(pipeline, sbt);
	}
}

size_t Model::GetMeshCount() const
{
	return m_Meshes.size();
}

const Mesh* Model::GetMesh(const size_t idx) const
{
	if (idx >= GetMeshCount()) { return nullptr; }
	return m_Meshes[idx];
}

const std::vector<Mesh*>& Model::GetMeshes() const
{
	return m_Meshes;
}

OptixTraversableHandle Model::GetTraversHandle() const
{
	return m_TraversHandle;
}

void Model::ProcessNode(aiNode* node, const aiScene* scene, const glm::mat4& parentT, const OptixDeviceContext optixDeviceContext)
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
		LoadMesh(mesh, scene, totalT, optixDeviceContext);
	}

	// Get children
	const size_t childCount = node->mNumChildren;
	//Log::Info("\tNode has " + std::to_string(childCount) + " children");
	for (size_t childIdx = 0; childIdx < childCount; ++childIdx)
	{
		ProcessNode(node->mChildren[childIdx], scene, totalT, optixDeviceContext);
	}
}

void Model::LoadMesh(aiMesh* mesh, const aiScene* scene, const glm::mat4& t, const OptixDeviceContext optixDeviceContext)
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

		// Get vert tangent
		glm::vec3 tangent(0.0f);
		if (mesh->HasTangentsAndBitangents())
		{
			tangent.x = mesh->mTangents[vertIdx].x;
			tangent.y = mesh->mTangents[vertIdx].y;
			tangent.z = mesh->mTangents[vertIdx].z;
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
		Vertex vert(pos, normal, tangent, uv);
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
	m_Meshes.push_back(new Mesh(vertices, indices, m_Materials[mesh->mMaterialIndex], optixDeviceContext));
}

void Model::LoadMaterials(const aiScene* scene, const SpecTexUsage specTexUsage)
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
		Log::Info("Diffuse color (" +
			std::to_string(diffColor.r) + ", " +
			std::to_string(diffColor.g) + ", " +
			std::to_string(diffColor.b) + ", " +
			std::to_string(diffColor.a) + ")");

		// Specular color
		aiColor4D aiSpecColor{};
		glm::vec4 specColor(1.0f);
		if (aiGetMaterialColor(material, AI_MATKEY_COLOR_SPECULAR, &aiSpecColor) == AI_SUCCESS)
		{
			specColor = { aiSpecColor.r, aiSpecColor.g, aiSpecColor.b, aiSpecColor.a };
		}
		Log::Info("Specular color (" +
			std::to_string(specColor.r) + ", " +
			std::to_string(specColor.g) + ", " +
			std::to_string(specColor.b) + ", " +
			std::to_string(specColor.a) + ")");

		// Roughness
		ai_real roughness = 1.0f;
		if (aiGetMaterialFloat(material, AI_MATKEY_ROUGHNESS_FACTOR, &roughness) != AI_SUCCESS)
		{
			roughness = 1.0f;
		}
		Log::Info("Roughness " + std::to_string(roughness));

		// Metalness
		ai_real metalness = 0.0f;
		if (aiGetMaterialFloat(material, AI_MATKEY_METALLIC_FACTOR, &metalness) != AI_SUCCESS)
		{
			metalness = 0.0f;
		}
		Log::Info("Metalness " + std::to_string(metalness));

		// Emissive
		aiColor4D aiEmissiveColor{};
		glm::vec4 emissiveColor(0.0f);
		if (aiGetMaterialColor(material, AI_MATKEY_COLOR_EMISSIVE, &aiEmissiveColor) == AI_SUCCESS)
		{
			emissiveColor = { aiEmissiveColor.r, aiEmissiveColor.g, aiEmissiveColor.b, aiEmissiveColor.a };
		}
		Log::Info("Emissive color (" +
			std::to_string(emissiveColor.r) + ", " +
			std::to_string(emissiveColor.g) + ", " +
			std::to_string(emissiveColor.b) + ", " +
			std::to_string(emissiveColor.a) + ")");

		// Diffuse texture
		const size_t diffTexCount = material->GetTextureCount(aiTextureType_DIFFUSE);
		Log::Info("Has " + std::to_string(diffTexCount) + " diffuse textures");
		Texture* diffTex = nullptr;
		if (diffTexCount > 0)
		{
			aiString texFileName{};
			material->GetTexture(aiTextureType_DIFFUSE, 0, &texFileName);
			Log::Info("Diffuse texture: " + std::string(texFileName.C_Str()));

			const std::string texFilePath = m_DirPath + "/" + texFileName.C_Str();
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

		// Specular texture
		const size_t specTexCount = material->GetTextureCount(aiTextureType_SPECULAR);
		Log::Info("Has " + std::to_string(specTexCount) + " specular textures");
		Texture* specTex = nullptr;
		if (specTexCount > 0)
		{
			aiString texFileName{};
			material->GetTexture(aiTextureType_SPECULAR, 0, &texFileName);
			Log::Info("Specular texture: " + std::string(texFileName.C_Str()));

			const std::string texFilePath = m_DirPath + "/" + texFileName.C_Str();
			if (m_Textures.find(texFilePath) == m_Textures.end())
			{
				specTex = new Texture(texFilePath);
				m_Textures.insert_or_assign(texFilePath, specTex);
			}
			else
			{
				specTex = m_Textures[texFilePath];
			}
		}

		// Roughness texture
		const size_t roughTexCount = material->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS);
		Log::Info("Has " + std::to_string(roughTexCount) + " roughness textures");
		Texture* roughTex = nullptr;
		if (roughTexCount > 0)
		{
			aiString texFileName{};
			material->GetTexture(aiTextureType_DIFFUSE_ROUGHNESS, 0, &texFileName);
			Log::Info("Roughness texture: " + std::string(texFileName.C_Str()));

			const std::string texFilePath = m_DirPath + "/" + texFileName.C_Str();
			if (m_Textures.find(texFilePath) == m_Textures.end())
			{
				roughTex = new Texture(texFilePath);
				m_Textures[texFilePath] = roughTex;
			}
			else
			{
				roughTex = m_Textures[texFilePath];
			}
		}

		// Roughness texture
		const size_t metalTexCount = material->GetTextureCount(aiTextureType_METALNESS);
		Log::Info("Has " + std::to_string(metalTexCount) + " metalness textures");
		Texture* metalTex = nullptr;
		if (metalTexCount > 0)
		{
			aiString texFileName{};
			material->GetTexture(aiTextureType_METALNESS, 0, &texFileName);
			Log::Info("Metalness texture: " + std::string(texFileName.C_Str()));

			const std::string texFilePath = m_DirPath + "/" + texFileName.C_Str();
			if (m_Textures.find(texFilePath) == m_Textures.end())
			{
				metalTex = new Texture(texFilePath);
				m_Textures[texFilePath] = metalTex;
			}
			else
			{
				metalTex = m_Textures[texFilePath];
			}
		}

		// Emissive texture
		const size_t emissiveTexCount = material->GetTextureCount(aiTextureType_EMISSION_COLOR);
		if (emissiveTexCount > 0)
		{
		}

		// Add material
		m_Materials.push_back(new Material(diffColor, specColor, emissiveColor, roughness, diffTex, specTex, roughTex, specTexUsage));
	}
}

void Model::BuildAccel(const OptixDeviceContext optixDeviceContext)
{
	// Construct array of OptixInstance
	const size_t meshCount = m_Meshes.size();
	std::vector<OptixInstance> instances(meshCount);

	for (size_t idx = 0; idx < meshCount; ++idx)
	{
		OptixInstance& instance = instances[idx];
		instance = {};
		instance.flags = OPTIX_INSTANCE_FLAG_NONE;
		instance.instanceId = idx;
		instance.sbtOffset = idx;
		instance.visibilityMask = 1;
		instance.traversableHandle = m_Meshes[idx]->GetTraversHandle();
		reinterpret_cast<glm::mat3x4&>(instance.transform) = glm::mat3x4(1.0f);
	}

	// Construct build input
	DeviceBuffer<OptixInstance> optixDevInstances(meshCount);
	optixDevInstances.Upload(instances.data());

	OptixBuildInput instanceInput{};
	instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	instanceInput.instanceArray.instances = optixDevInstances.GetCuPtr();
	instanceInput.instanceArray.numInstances = meshCount;

	// Build options
	OptixAccelBuildOptions buildOptions{};
	buildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	buildOptions.motionOptions.numKeys = 1;
	buildOptions.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;
	buildOptions.motionOptions.timeBegin = 0.0f;
	buildOptions.motionOptions.timeEnd = 1.0f;
	buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	// Get memory usage
	OptixAccelBufferSizes accelBufferSizes{};
	ASSERT_OPTIX(optixAccelComputeMemoryUsage(optixDeviceContext, &buildOptions, &instanceInput, 1, &accelBufferSizes));

	DeviceBuffer<uint64_t> compactSizeBuf(1);
	DeviceBuffer<uint8_t> tempBuffer(accelBufferSizes.tempSizeInBytes);
	DeviceBuffer<uint8_t> outputBuffer(accelBufferSizes.outputSizeInBytes);

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactSizeBuf.GetCuPtr();

	// Build
	ASSERT_OPTIX(optixAccelBuild(
		optixDeviceContext,
		0,
		&buildOptions,
		&instanceInput,
		1,
		tempBuffer.GetCuPtr(),
		tempBuffer.GetByteSize(),
		outputBuffer.GetCuPtr(),
		outputBuffer.GetByteSize(),
		&m_TraversHandle,
		&emitDesc,
		1));

	// Sync
	ASSERT_CUDA(cudaDeviceSynchronize());

	// Compact
	uint64_t compactedSize = 0;
	compactSizeBuf.Download(&compactedSize);

	m_AccelBuf.Alloc(compactedSize);
	ASSERT_OPTIX(optixAccelCompact(
		optixDeviceContext,
		0,
		m_TraversHandle,
		m_AccelBuf.GetCuPtr(),
		m_AccelBuf.GetByteSize(),
		&m_TraversHandle));

	// Sync
	ASSERT_CUDA(cudaDeviceSynchronize());
}
