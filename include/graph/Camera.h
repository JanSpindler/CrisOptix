#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

struct CameraData
{
	glm::vec3 pos;
	glm::vec3 U, V, W;
	glm::mat4 currW2V;
	glm::mat4 prevW2V;
};

static __forceinline__ __host__ __device__ void SpawnCameraRay(
	const CameraData& camData,
	const glm::vec2& viewportCoord,
	glm::vec3& rayOrigin,
	glm::vec3& rayDir)
{
	rayOrigin = camData.pos;
	rayDir = glm::normalize(camData.U * viewportCoord.x + camData.V * viewportCoord.y + camData.W);
}

class Camera
{
public:
	Camera(
		const glm::vec3& pos, 
		const glm::vec3& viewDir, 
		const glm::vec3& up, 
		const float aspectRatio, 
		const float fov);

	CameraData GetData() const;

	void RotateAroundOrigin(const glm::vec3& axis, float angle);

	void Move(const glm::vec3& move);
	void RotateViewDir(float phi, float theta);

	const glm::vec3& GetPos() const;
	void SetPos(const glm::vec3& pos);

	const glm::vec3& GetViewDir() const;
	void SetViewDir(const glm::vec3& viewDir);

	const glm::vec3& GetUp() const;
	void SetUp(const glm::vec3& up);

	bool HasChanged() const;
	void SetChanged(bool changed);

	float GetAspectRatio() const;
	void SetAspectRatio(float aspectRatio);
	void SetAspectRatio(uint32_t width, uint32_t height);

	float GetFov() const;
	void SetFov(float fov);

	float GetNearPlane() const;
	void SetNearPlane(float nearPlane);

	float GetFarPlane() const;
	void SetFarPlane(float farPlane);

private:
	glm::vec3 m_Pos{};
	glm::vec3 m_ViewDir{};
	glm::vec3 m_Up{};

	bool m_Changed = false;

	float m_AspectRatio = 1.0f;
	float m_Fov = glm::radians(60.0f);
	float m_NearPlane = 0.01f;
	float m_FarPlane = 100.0f;

	mutable CameraData m_CamData{};
};
