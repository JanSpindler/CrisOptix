#pragma once

#include <graph/Interaction.h>
#include <graph/LaunchParams.h>
#include <graph/restir/PrefixPath.h>
#include <graph/restir/SuffixPath.h>
#include <graph/restir/Reconnection.h>
#include <graph/sample_emitter.h>
#include <optix_device.h>

static __forceinline__ __device__ float GetReconnectionJacobian(
	const glm::vec3& oldXi, 
	const glm::vec3& newXi, 
	const glm::vec3& targetPos, 
	const glm::vec3& targetNormal)
{
	// TODO: Implement for random replay
	
	const float term1 = glm::distance(targetPos, oldXi) / glm::distance(targetPos, newXi);
	const float term2 = glm::dot(newXi - targetPos, targetNormal) / glm::dot(oldXi - targetPos, targetNormal);
	return term1 * term2;
}

// Returns true if reconnection is valid (no occlusion / valid brdf pdf)
static __forceinline__ __device__ bool CalcReconnection(
	Reconnection& recon,
	const PrefixPath& prefix,
	const SuffixPath& suffix,
	const LaunchParams& params)
{
	// TODO: Hybrid shift / right now only pure reconnection
	
	//
	const glm::vec3 lastPrefixPos = prefix.lastInteraction.pos;
	const glm::vec3 reconVector = suffix.firstPos - lastPrefixPos;
	const glm::vec3 reconDir = glm::normalize(reconVector);

	// Check occlusion between last prefix vert and first suffix vert
	// Cast shadow ray
	const bool occluded = TraceOcclusion(
		params.traversableHandle,
		lastPrefixPos,
		reconDir,
		1e-3f,
		glm::length(reconVector),
		params.occlusionTraceParams);
	if (occluded) { return false; }

	// First brdf sample at last prefix vert
	const BrdfEvalResult pos0Eval = optixDirectCall<BrdfEvalResult, const SurfaceInteraction&, const glm::vec3&>(
		prefix.lastInteraction.meshSbtData->evalMaterialSbtIdx,
		prefix.lastInteraction,
		reconDir);
	if (pos0Eval.samplingPdf <= 0.0f) { return false; }

	// Construct pos0Brdf from pos0Eval
	recon.pos0Brdf.outDir = reconDir;
	recon.pos0Brdf.roughness = pos0Eval.roughness;
	recon.pos0Brdf.samplingPdf = pos0Eval.samplingPdf; // TODO: samplingPdf is currently always 1
	recon.pos0Brdf.brdfVal = pos0Eval.brdfResult;

	// Retrace ray to first suffix vert
	// Sample surface interaction
	SurfaceInteraction interaction{};
	TraceWithDataPointer<SurfaceInteraction>(
		params.traversableHandle,
		lastPrefixPos,
		reconDir,
		1e-3f,
		1e16f,
		params.surfaceTraceParams,
		&interaction);
	if (!interaction.valid) { return false; }
	
	// Check if retrace found the same first suffix vert
	if (glm::distance(interaction.pos, suffix.firstPos) > 0.01f) { return false; }

	// Second brdf sample at first suffix vert
	const BrdfEvalResult pos1Eval = optixDirectCall<BrdfEvalResult, const SurfaceInteraction&, const glm::vec3&>(
		interaction.meshSbtData->evalMaterialSbtIdx,
		interaction,
		suffix.firstDir);

	// Construct pos1Brdf from pos1Eval
	recon.pos1Brdf.outDir = reconDir;
	recon.pos1Brdf.roughness = pos1Eval.roughness;
	recon.pos1Brdf.samplingPdf = pos1Eval.samplingPdf; // TODO: samplingPdf is currently always 1
	recon.pos1Brdf.brdfVal = pos1Eval.brdfResult;

	//
	return true;
}

static __forceinline__ __device__ void GenPrefix(
	PrefixPath& prefix,
	const glm::vec3& origin,
	const glm::vec3& dir,
	const size_t maxLen,
	const size_t maxNeeTries,
	PCG32& rng,
	const LaunchParams& params)
{
	glm::vec3 currentPos = origin;
	glm::vec3 currentDir = dir;
	size_t currentDepth = 0;

	prefix = {};
	prefix.valid = true;
	prefix.nee = false;
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

		// Decide if NEE or continue PT
		if (rng.NextFloat() < params.neeProb)
		{
			// NEE
			bool validEmitterFound = false;
			size_t neeCounter = 0;
			while (!validEmitterFound && neeCounter < maxNeeTries)
			{
				//
				++neeCounter;

				// Sample light source
				const EmitterSample emitterSample = SampleEmitter(rng, params.emitterTable);
				const glm::vec3 lightDir = glm::normalize(emitterSample.pos - prefix.lastInteraction.pos);
				const float distance = glm::length(emitterSample.pos - prefix.lastInteraction.pos);

				// Cast shadow ray
				const bool occluded = TraceOcclusion(
					params.traversableHandle,
					prefix.lastInteraction.pos,
					lightDir,
					1e-3f,
					distance,
					params.occlusionTraceParams);

				// If emitter is occluded -> skip
				if (occluded)
				{
					prefix.nee = false;
					validEmitterFound = false;
				}
				// If emitter is not occluded -> end NEE
				else
				{
					// Calc brdf
					const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const SurfaceInteraction&, const glm::vec3&>(
						prefix.lastInteraction.meshSbtData->evalMaterialSbtIdx,
						prefix.lastInteraction,
						lightDir);
					prefix.throughput *= brdfEvalResult.brdfResult * emitterSample.color;
					if (currentDepth == 1) { prefix.throughput += brdfEvalResult.emission; }

					prefix.nee = true;
					validEmitterFound = true;
				}
			}

			if (!validEmitterFound)
			{
				prefix.nee = false;
			}

			break;
		}

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
		prefix.throughput *= brdfSampleResult.brdfVal / brdfSampleResult.samplingPdf;
		prefix.p *= brdfSampleResult.samplingPdf;
	}

	prefix.len = currentDepth;
}

static __forceinline__ __device__ void GenSuffix(
	SuffixPath& suffix,
	const glm::vec3& origin,
	const glm::vec3& dir,
	const size_t maxLen,
	const size_t maxNeeTries,
	PCG32& rng,
	const LaunchParams& params)
{
	glm::vec3 currentPos = origin;
	glm::vec3 currentDir = dir;
	size_t currentDepth = 0;

	suffix.valid = true;
	suffix.firstPos = origin;
	suffix.firstDir = dir;
	suffix.len = 0;
	suffix.p = 1.0f;
	suffix.throughput = glm::vec3(0.0f);
	suffix.rng = rng;

	glm::vec3 pathThroughput(1.0f);

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
		if (rng.NextFloat() < params.neeProb)
		{
			// NEE
			bool validEmitterFound = false;
			size_t neeCounter = 0;
			while (!validEmitterFound && neeCounter < maxNeeTries)
			{
				//
				++neeCounter;

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
					suffix.throughput = pathThroughput* brdfEvalResult.brdfResult * emitterSample.color;
					if (currentDepth == 1) { suffix.throughput += brdfEvalResult.emission; }

					suffix.p *= emitterSample.p;
					
					validEmitterFound = true;
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
		pathThroughput *= brdfSampleResult.brdfVal / brdfSampleResult.samplingPdf;
		suffix.p *= brdfSampleResult.samplingPdf;
	}

	suffix.len = currentDepth;
}

static __forceinline__ __device__ glm::vec3 TraceCompletePath(
	const glm::vec3& origin,
	const glm::vec3& dir,
	const size_t maxLen,
	const size_t maxNeeTries,
	PCG32& rng,
	const LaunchParams& params)
{
	glm::vec3 currentPos = origin;
	glm::vec3 currentDir = dir;
	size_t currentDepth = 0;

	glm::vec3 radiance(0.0f);
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
			break;
		}

		//
		++currentDepth;

		// Decide if NEE or continue PT
		if (rng.NextFloat() < params.neeProb)
		{
			// NEE
			bool validEmitterFound = false;
			size_t neeCounter = 0;
			while (!validEmitterFound && neeCounter < maxNeeTries)
			{
				//
				++neeCounter;

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
					radiance = throughput * brdfEvalResult.brdfResult * emitterSample.color;
					if (currentDepth == 1) { radiance += brdfEvalResult.emission; }

					validEmitterFound = true;
				}
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
			break;
		}

		currentPos = interaction.pos;
		currentDir = brdfSampleResult.outDir;
		throughput *= brdfSampleResult.brdfVal / brdfSampleResult.samplingPdf;
	}

	return radiance;
}
