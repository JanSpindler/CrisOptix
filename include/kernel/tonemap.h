#pragma once

#include <CuBufferView.h>
#include <glm/glm.hpp>

void ToneMapping(const CuBufferView<glm::vec3>& inputHdr, CuBufferView<glm::u8vec3>& outputLdr);
