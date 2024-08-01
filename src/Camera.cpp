#include <graph/Camera.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>

Camera::Camera(
	const glm::vec3& pos,
	const glm::vec3& viewDir,
	const glm::vec3& up,
	const float aspectRatio,
	const float fov)
	:
	m_Pos(pos),
	m_ViewDir(viewDir),
	m_Up(up),
	m_AspectRatio(aspectRatio),
	m_Fov(fov)
{
}

CameraData Camera::GetData() const
{
	m_CamData.prevW2V = m_CamData.currW2V;
	m_CamData.currW2V = glm::lookAt(m_Pos, m_Pos + m_ViewDir, m_Up);
	const glm::mat4 toWorld = glm::inverse(m_CamData.currW2V);

	m_CamData.pos = m_Pos;
	m_CamData.U = glm::normalize(glm::vec3(toWorld[0])) * glm::tan(0.5f * m_Fov) * m_AspectRatio;
	m_CamData.V = -glm::normalize(glm::vec3(toWorld[1])) * glm::tan(0.5f * m_Fov);
	m_CamData.W = -glm::normalize(glm::vec3(toWorld[2]));

	return m_CamData;
}

void Camera::RotateAroundOrigin(const glm::vec3& axis, float angle)
{
	m_Pos = glm::rotate(angle, axis) * glm::vec4(m_Pos, 1.0);
	m_ViewDir = -glm::normalize(m_Pos);
}

void Camera::Move(const glm::vec3& move)
{
	glm::vec3 frontMove = glm::normalize(m_ViewDir * glm::vec3(1.0f, 0.0f, 1.0f)) * move.z;
	glm::vec3 sideMove = glm::normalize(glm::cross(m_ViewDir, m_Up)) * move.x;
	glm::vec3 upMove(0.0f, move.y, 0.0f);
	m_Pos += frontMove + sideMove + upMove;
}

void Camera::RotateViewDir(float phi, float theta)
{
	glm::vec3 phiAxis = m_Up;
	glm::mat3 phiMat = glm::rotate(glm::identity<glm::mat4>(), phi, phiAxis);

	glm::vec3 thetaAxis = glm::normalize(glm::cross(m_ViewDir, m_Up));
	glm::mat3 thetaMat = glm::rotate(glm::identity<glm::mat4>(), theta, thetaAxis);

	m_ViewDir = glm::normalize(thetaMat * phiMat * m_ViewDir);
}

const glm::vec3& Camera::GetPos() const
{
	return m_Pos;
}

void Camera::SetPos(const glm::vec3& pos)
{
	m_Pos = pos;
}

const glm::vec3& Camera::GetViewDir() const
{
	return m_ViewDir;
}

void Camera::SetViewDir(const glm::vec3& viewDir)
{
	m_ViewDir = glm::normalize(viewDir);
}

const glm::vec3& Camera::GetUp() const
{
	return m_Up;
}

void Camera::SetUp(const glm::vec3& up)
{
	m_Up = glm::normalize(up);
}

bool Camera::HasChanged() const
{
	return m_Changed;
}

void Camera::SetChanged(bool changed)
{
	m_Changed = changed;
}

float Camera::GetAspectRatio() const
{
	return m_AspectRatio;
}

void Camera::SetAspectRatio(float aspectRatio)
{
	m_AspectRatio = aspectRatio;
}

void Camera::SetAspectRatio(uint32_t width, uint32_t height)
{
	if (height == 0)
		height = 1;
	m_AspectRatio = static_cast<float>(width) / static_cast<float>(height);
}

float Camera::GetFov() const
{
	return m_Fov;
}

void Camera::SetFov(float fov)
{
	m_Fov = fov;
}

float Camera::GetNearPlane() const
{
	return m_NearPlane;
}

void Camera::SetNearPlane(float nearPlane)
{
	m_NearPlane = nearPlane;
}

float Camera::GetFarPlane() const
{
	return m_FarPlane;
}

void Camera::SetFarPlane(float farPlane)
{
	m_FarPlane = farPlane;
}
