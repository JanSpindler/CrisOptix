#pragma once

#include <glm/glm.hpp>
#include <model/Model.h>

class ModelInstance
{
public:
	ModelInstance(const Model& model, const glm::mat4& modelMat);

	const Model& GetModel() const;
	const glm::mat4& GetModelMat() const;
	void SetModelMat(const glm::mat4& modelMat);

private:
	const Model& m_Model;
	glm::mat4 m_ModelMat{};
};

