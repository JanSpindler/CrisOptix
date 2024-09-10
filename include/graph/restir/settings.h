#pragma once

#include <cstdint>

enum class RendererType : int
{
	PathTracer,
	RestirPt,
	ConditionalRestir,
};

static constexpr size_t RENDERER_TYPE_COUNT = 3;
static constexpr const char* RENDERER_TYPE_NAMES[] = { "Path Tracer", "Restir Pt", "Conditional Restir" };

enum class PrefixRadiusType : int
{
	Constant,
	PathLength
};

static constexpr size_t PREFIX_RADIUS_TYPE_COUNT = 2;
static constexpr const char* PREFIX_RADIUS_TYPE_NAMES[] = { "Constant", "Path Length" };
