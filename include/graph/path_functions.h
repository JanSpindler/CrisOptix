#pragma once

#include <graph/Path.h>
#include <graph/LaunchParams.h>
#include <util/random.h>
#include <graph/trace.h>
#include <graph/Interaction.h>
#include <graph/sample_emitter.h>
#include <optix_device.h>
#include <limits>
#include <cuda/std/limits>

static constexpr size_t MAX_TRACE_OPS = 16;
static constexpr float NEE_PROB = 0.5f;

static __forceinline__ __device__ Path SamplePrefix(const glm::vec3& origin, const glm::vec3& dir, PCG32& rng, const LaunchParams& params, glm::vec3& outDir)
{
	glm::vec3 currentPos = origin;
	glm::vec3 currentDir = dir;
	size_t currentDepth = 0;
	
	Path path{};
	path.vertices[0] = currentPos;
	path.throughput = glm::vec3(1.0f);
	path.outputRadiance = glm::vec3(0.0f);

	// Trace
	for (uint32_t traceIdx = 0; traceIdx < params.restirParams.prefixLength; ++traceIdx)
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
		if (!interaction.valid) { break; }

		//
		++currentDepth;
		path.vertices[currentDepth] = interaction.pos;

		// Indirect illumination, generate next ray
		const glm::vec3 brdfRand = rng.Next3d();
		path.randomVars[currentDepth][0].randFloat = brdfRand[0];
		path.randomVars[currentDepth][1].randFloat = brdfRand[1];
		path.randomVars[currentDepth][2].randFloat = brdfRand[2];

		const BrdfSampleResult brdfSampleResult = optixDirectCall<BrdfSampleResult, const SurfaceInteraction&, const glm::vec3&>(
			interaction.meshSbtData->sampleMaterialSbtIdx,
			interaction,
			brdfRand);
		if (brdfSampleResult.samplingPdf <= 0.0f) { break; }

		currentPos = interaction.pos;
		currentDir = brdfSampleResult.outDir;
		path.throughput *= brdfSampleResult.weight;
	}

	outDir = currentDir;

	path.prefixLength = currentDepth;
	path.length = currentDepth;
	path.emitterSample.p = cuda::std::numeric_limits<float>::infinity();
	return path;
}

static __forceinline__ __device__ Path SamplePath(const glm::vec3& origin, const glm::vec3& dir, const size_t maxLen, PCG32& rng, const LaunchParams& params)
{
	glm::vec3 currentPos = origin;
	glm::vec3 currentDir = dir;
	size_t currentDepth = 0;

	Path path{};
	path.vertices[0] = currentPos;
	path.throughput = glm::vec3(1.0f);
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
		if (!interaction.valid) { break; }

		//
		++currentDepth;
		path.vertices[currentDepth] = interaction.pos;

		// Decide if NEE or continue PT
		if (rng.NextFloat() < NEE_PROB || currentDepth >= maxLen)
		{
			// NEE
			// Sample light source
			const EmitterSample emitterSample = SampleEmitter(rng, params.emitterTable);
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
			path.outputRadiance = path.throughput * brdfEvalResult.brdfResult * emitterSample.color;
			if (currentDepth == 1) { path.outputRadiance += brdfEvalResult.emission; }

			// Exit from PT
			break;
		}

		// Indirect illumination, generate next ray
		const BrdfSampleResult brdfSampleResult = optixDirectCall<BrdfSampleResult, const SurfaceInteraction&, PCG32&>(
			interaction.meshSbtData->sampleMaterialSbtIdx,
			interaction,
			rng);
		if (brdfSampleResult.samplingPdf <= 0.0f) { break; }

		currentPos = interaction.pos;
		currentDir = brdfSampleResult.outDir;
		path.throughput *= brdfSampleResult.weight;
	}

	path.prefixLength = 0;
	path.length = currentDepth;
	return path;
}

static __forceinline__ __device__ glm::vec3 EvalPath(const Path& path)
{

}
