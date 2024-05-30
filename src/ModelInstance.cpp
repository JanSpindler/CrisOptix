#include <model/ModelInstance.h>

ModelInstance::ModelInstance(const Model* model, const glm::mat4& modelMat) :
	m_Model(model),
	m_ModelMat(modelMat)
{
}

const Model* ModelInstance::GetModel() const
{
	return m_Model;
}

const glm::mat4& ModelInstance::GetModelMat() const
{
	return m_ModelMat;
}

void ModelInstance::SetModelMat(const glm::mat4& modelMat)
{
	m_ModelMat = modelMat;
}
