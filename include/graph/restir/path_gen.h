#pragma once

#include <graph/Interaction.h>
#include <graph/restir/struct/PathState.h>
#include <graph/LaunchParams.h>
#include <graph/restir/struct/PrefixPath.h>
#include <graph/restir/struct/SuffixPath.h>
#include <graph/restir/struct/Reconnection.h>

static __forceinline__ __device__ void GenPrefix(
	PrefixPath& prefix,
	const glm::vec3& origin,
	const glm::vec3& dir,
	const size_t maxLen,
	PCG32& rng,
	const LaunchParams& params)
{
	glm::vec3 currentPos = origin;
	glm::vec3 currentDir = dir;
	size_t currentDepth = 0;

	prefix = {};
	prefix.valid = true;
	prefix.len = 0;
	prefix.p = 1.0f;
	prefix.throughput = glm::vec3(1.0f);

	// Trace
	for (uint32_t traceIdx = 0; traceIdx < maxLen; ++traceIdx)
	{
		// Sample surface interaction
		TraceWithDataPointer<SurfaceInteraction>(
			params.traversableHandle,
			currentPos,
			currentDir,
			1e-3f,
			1e16f,
			params.surfaceTraceParams,
			&prefix.lastInteraction);

		// Exit if no surface found
		if (!prefix.lastInteraction.valid)
		{
			prefix.valid = false;
			break;
		}

		//
		++currentDepth;

		// Do not sample brdf if last vertex
		if (traceIdx == maxLen - 1) { break; }

		// Indirect illumination, generate next ray
		const BrdfSampleResult brdfSampleResult = optixDirectCall<BrdfSampleResult, const SurfaceInteraction&, PCG32&>(
			prefix.lastInteraction.meshSbtData->sampleMaterialSbtIdx,
			prefix.lastInteraction,
			rng);
		if (brdfSampleResult.samplingPdf <= 0.0f)
		{
			prefix.valid = false;
			break;
		}

		currentPos = prefix.lastInteraction.pos;
		currentDir = brdfSampleResult.outDir;
		prefix.throughput *= brdfSampleResult.weight;
		prefix.p *= brdfSampleResult.samplingPdf;
	}

	prefix.len = currentDepth;
}

static __forceinline__ __device__ void GenSuffix(
	SuffixPath& suffix,
	const glm::vec3& origin,
	const glm::vec3& dir,
	const size_t maxLen,
	const float neeProb,
	const size_t maxNeeTries,
	PCG32& rng,
	const LaunchParams& params)
{
	glm::vec3 currentPos = origin;
	glm::vec3 currentDir = dir;
	size_t currentDepth = 0;

	suffix = {};
	suffix.valid = true;
	suffix.firstPos = origin;
	suffix.len = 0;
	suffix.p = 1.0f;
	suffix.radiance = glm::vec3(0.0f);
	suffix.rng = rng;

	glm::vec3 throughput(1.0f);

	// Trace
	for (uint32_t traceIdx = 0; traceIdx < maxLen; ++traceIdx)
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
		if (!interaction.valid)
		{
			suffix.valid = false;
			break;
		}

		//
		++currentDepth;

		// Decide if NEE or continue PT
		if (rng.NextFloat() < neeProb)
		{
			// NEE
			bool validEmitterFound = false;
			size_t neeCounter = 0;
			while (!validEmitterFound && neeCounter < maxNeeTries)
			{
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

				// If emitter is occluded -> skip
				if (occluded)
				{
					validEmitterFound = false;
				}
				// If emitter is not occluded -> end NEE
				else
				{
					// Calc brdf
					const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const SurfaceInteraction&, const glm::vec3&>(
						interaction.meshSbtData->evalMaterialSbtIdx,
						interaction,
						lightDir);
					suffix.radiance = throughput * brdfEvalResult.brdfResult * emitterSample.color;
					if (currentDepth == 1) { suffix.radiance += brdfEvalResult.emission; }

					suffix.p *= emitterSample.p;
					//path.emitter = emitterSample;
				}
			}

			if (!validEmitterFound)
			{
				suffix.valid = false;
			}

			break;
		}

		// Indirect illumination, generate next ray
		const BrdfSampleResult brdfSampleResult = optixDirectCall<BrdfSampleResult, const SurfaceInteraction&, PCG32&>(
			interaction.meshSbtData->sampleMaterialSbtIdx,
			interaction,
			rng);
		if (brdfSampleResult.samplingPdf <= 0.0f)
		{
			suffix.valid = false;
			break; 
		}

		currentPos = interaction.pos;
		currentDir = brdfSampleResult.outDir;
		throughput *= brdfSampleResult.weight;
		suffix.p *= brdfSampleResult.samplingPdf;
	}

	suffix.len = currentDepth;
}
