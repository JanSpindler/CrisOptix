#pragma once

#include <graph/Path.h>
#include <graph/LaunchParams.h>
#include <util/random.h>
#include <graph/trace.h>
#include <graph/Interaction.h>
#include <graph/sample_emitter.h>
#include <optix_device.h>

static constexpr size_t MAX_TRACE_OPS = 8;
static constexpr size_t MAX_TRACE_DEPTH = 8;
static constexpr float NEE_PROB = 0.5f;

static constexpr __device__ Path SamplePath(const glm::vec3& origin, const glm::vec3& dir, PCG32& rng, LaunchParams& params)
{
	glm::vec3 currentPos = origin;
	glm::vec3 currentDir = dir;
	size_t currentDepth = 0;
	glm::vec3 currentThroughput(1.0f);

	Path path{};
	path.vertices[0] = currentPos;
	path.outputRadiance = glm::vec3(0.0f);

	// Trace
	for (uint32_t traceIdx = 0; traceIdx < MAX_TRACE_OPS; ++traceIdx)
	{
		// Sample surface interaction
		SurfaceInteraction interaction{};
		TraceWithDataPointer<SurfaceInteraction>(
			params.traversableHandle,
			currentPos,
			currentDir,
			1e-3f,
			1e16f,
			params.surfaceTraceParams,
			&interaction);

		// Exit if no surface found
		if (!interaction.valid) { continue; }

		//
		++currentDepth;
		path.vertices[currentDepth] = interaction.pos;

		// Decide if NEE or continue PT
		if (rng.NextFloat() < NEE_PROB || currentDepth >= MAX_TRACE_DEPTH)
		{
			// NEE
			// Sample light source
			const EmitterSample emitterSample = SampleEmitter(rng, params);
			const glm::vec3 lightDir = glm::normalize(emitterSample.pos - interaction.pos);
			const float distance = glm::length(emitterSample.pos - interaction.pos);

			// Cast shadow ray
			const bool occluded = TraceOcclusion(
				params.traversableHandle,
				interaction.pos,
				lightDir,
				1e-3f,
				distance,
				params.occlusionTraceParams);
			if (occluded) { continue; }

			// Calc brdf
			const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const SurfaceInteraction&, const glm::vec3&>(
				interaction.meshSbtData->evalMaterialSbtIdx,
				interaction,
				lightDir);
			path.outputRadiance = currentThroughput * brdfEvalResult.brdfResult * emitterSample.color;
			if (currentDepth == 1) { path.outputRadiance += brdfEvalResult.emission; }

			// Exit from PT
			break;
		}

		// Indirect illumination, generate next ray
		BrdfSampleResult brdfSampleResult = optixDirectCall<BrdfSampleResult, const SurfaceInteraction&, PCG32&>(
			interaction.meshSbtData->sampleMaterialSbtIdx,
			interaction,
			rng);
		if (brdfSampleResult.samplingPdf <= 0.0f) { break; }

		currentPos = interaction.pos;
		currentDir = brdfSampleResult.outDir;
		currentThroughput *= brdfSampleResult.weight;
	}

	return path;
}
