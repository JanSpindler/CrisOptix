#pragma once

#include <cuda_runtime.h>
#include <graph/Camera.h>

__host__ __device__ void SpawnCameraRay(
	const CameraData& camData, 
	const glm::vec2& viewportCoord, 
	glm::vec3& rayOrigin, 
	glm::vec3& rayDir)
{
	rayOrigin = camData.pos;
	rayDir = glm::normalize(camData.U * viewportCoord.x + camData.V * viewportCoord.y + camData.W);
}
