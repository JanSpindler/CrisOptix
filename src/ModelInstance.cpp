#include <model/ModelInstance.h>

ModelInstance::ModelInstance(const Model& model, const glm::mat4& transform) :
	m_Model(model),
	m_Transform(transform)
{
}

const Model& ModelInstance::GetModel() const
{
	return m_Model;
}

const glm::mat4& ModelInstance::GetTransform() const
{
	return m_Transform;
}

void ModelInstance::SetTransform(const glm::mat4& transform)
{
	m_Transform = transform;
}
