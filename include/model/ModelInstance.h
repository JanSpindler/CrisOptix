#pragma once

#include <glm/glm.hpp>
#include <model/Model.h>

class ModelInstance
{
public:
	ModelInstance(const Model& model, const glm::mat4& transform);

	const Model& GetModel() const;
	const glm::mat4& GetTransform() const;
	void SetTransform(const glm::mat4& transform);

private:
	const Model& m_Model;
	glm::mat4 m_Transform{};
};

