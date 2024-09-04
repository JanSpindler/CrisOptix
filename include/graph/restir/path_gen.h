#pragma once

#include <graph/Interaction.h>
#include <graph/LaunchParams.h>
#include <graph/restir/PrefixPath.h>
#include <graph/restir/SuffixPath.h>
#include <graph/restir/Reconnection.h>
#include <graph/sample_emitter.h>
#include <optix_device.h>
#include <graph/trace.h>

static __forceinline__ __device__ float CalcReconnectionJacobian(
	const glm::vec3& oldXi, 
	const glm::vec3& newXi, 
	const glm::vec3& targetPos, 
	const glm::vec3& targetNormal)
{
	const float term1 = glm::dot(newXi - targetPos, targetNormal) / glm::dot(oldXi - targetPos, targetNormal);
	const float term2 = glm::distance(targetPos, oldXi) / glm::distance(targetPos, newXi);
	return term1 * term2;
}

static __forceinline__ __device__ PrefixPath TracePrefix(
	const glm::vec3& origin,
	const glm::vec3& dir,
	const size_t maxLen,
	const size_t maxNeeTries,
	Interaction& primaryInteraction,
	PCG32& rng,
	const LaunchParams& params)
{
	glm::vec3 currentPos = origin;
	glm::vec3 currentDir = dir;

	PrefixPath prefix{};
	prefix.rng = rng;
	prefix.p = 1.0f;
	prefix.f = glm::vec3(1.0f);
	prefix.postReconF = glm::vec3(1.0f);
	prefix.SetValid(true);

	// Trace
	for (uint32_t traceIdx = 0; traceIdx < maxLen; ++traceIdx)
	{
		// Sample surface interaction
		TraceWithDataPointer<Interaction>(
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
			prefix.SetValid(false);
			break;
		}

		//
		prefix.SetLength(prefix.GetLength() + 1);
		
		//
		if (prefix.GetLength() == 1)
		{
			prefix.primaryHitPos = prefix.lastInteraction.pos; 
			prefix.primaryHitInDir = dir;
			primaryInteraction = prefix.lastInteraction;
		}

		// TODO: Also include roughness
		const bool postRecon = prefix.GetLength() > 1;

		// Decide if NEE or continue PT
		if (rng.NextFloat() < params.neeProb)
		{
			//
			prefix.p *= params.neeProb;

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
					validEmitterFound = false;
				}
				// If emitter is not occluded -> end NEE
				else
				{
					// Calc brdf
					const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
						prefix.lastInteraction.meshSbtData->evalMaterialSbtIdx,
						prefix.lastInteraction,
						lightDir);
					
					prefix.f *= brdfEvalResult.brdfResult * emitterSample.color;
					if (postRecon) { prefix.postReconF *= brdfEvalResult.brdfResult * emitterSample.color; }

					if (prefix.GetLength() == 1) { prefix.f += brdfEvalResult.emission; }

					prefix.SetNee(true);
					prefix.SetValid(true);

					validEmitterFound = true;
				}
			}

			if (!validEmitterFound)
			{
				prefix.SetNee(false);
				prefix.SetValid(false);
			}

			break;
		}

		// Store as reconnection vertex if fit
		if (postRecon && prefix.GetReconIdx() == 0)
		{
			prefix.reconInteraction = prefix.lastInteraction;
			prefix.SetReconIdx(prefix.GetLength());
		}

		// Do not sample brdf when at last position
		if (prefix.GetLength() == maxLen) { break; }

		// Indirect illumination, generate next ray
		const BrdfSampleResult brdfSampleResult = optixDirectCall<BrdfSampleResult, const Interaction&, PCG32&>(
			prefix.lastInteraction.meshSbtData->sampleMaterialSbtIdx,
			prefix.lastInteraction,
			rng);
		if (brdfSampleResult.samplingPdf <= 0.0f)
		{
			prefix.SetValid(false);
			break;
		}

		currentPos = prefix.lastInteraction.pos;
		currentDir = brdfSampleResult.outDir;

		prefix.f *= brdfSampleResult.brdfVal;
		if (postRecon) { prefix.postReconF *= brdfSampleResult.brdfVal; }
		prefix.p *= brdfSampleResult.samplingPdf * (1.0f - params.neeProb);
	}

	return prefix;
}

static __forceinline__ __device__ SuffixPath TraceSuffix(
	const PrefixPath& prefix,
	const size_t maxLen,
	const size_t maxNeeTries,
	PCG32& rng,
	const LaunchParams& params)
{
	SuffixPath suffix{};
	suffix.f = glm::vec3(1.0f);
	suffix.p = 1.0f;
	suffix.postReconF = glm::vec3(1.0f);
	suffix.rng = rng;
	suffix.lastPrefixPos = prefix.lastInteraction.pos;
	suffix.lastPrefixInDir = prefix.lastInteraction.inRayDir;

	Interaction interaction{};

	// Suffix may directly terminate by NEE
	if (rng.NextFloat() < params.neeProb)
	{
		//
		suffix.p *= params.neeProb;

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
				validEmitterFound = false;
			}
			// If emitter is not occluded -> end NEE
			else
			{
				// Trace surface interaction at emitter to store as reconInteraction
				TraceWithDataPointer<Interaction>(
					params.traversableHandle,
					prefix.lastInteraction.pos,
					lightDir,
					1e-3f,
					distance + 1.0f,
					params.surfaceTraceParams,
					&interaction);
				suffix.reconInteraction = interaction;

				// Calc brdf
				const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
					prefix.lastInteraction.meshSbtData->evalMaterialSbtIdx,
					prefix.lastInteraction,
					lightDir);

				suffix.f *= brdfEvalResult.brdfResult * emitterSample.color;
				suffix.postReconF = suffix.f;

				suffix.SetValid(false);
				validEmitterFound = true;
			}
		}

		if (!validEmitterFound) { suffix.SetValid(false); }
		return suffix;
	}

	// If not directly terminated by NEE -> Sample direction from brdf at last vertex of prefix
	const BrdfSampleResult brdfSampleResult = optixDirectCall<BrdfSampleResult, const Interaction&, PCG32&>(
		prefix.lastInteraction.meshSbtData->sampleMaterialSbtIdx,
		prefix.lastInteraction,
		rng);
	if (brdfSampleResult.samplingPdf <= 0.0f)
	{
		suffix.SetValid(false);
		return suffix;
	}

	glm::vec3 currentPos = prefix.lastInteraction.pos;
	glm::vec3 currentDir = brdfSampleResult.outDir;
	suffix.f *= brdfSampleResult.brdfVal;
	suffix.p *= brdfSampleResult.samplingPdf * (1.0f - params.neeProb);

	// Trace
	for (uint32_t traceIdx = 0; traceIdx < maxLen; ++traceIdx)
	{
		// Sample surface interaction
		TraceWithDataPointer<Interaction>(
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
			suffix.SetValid(false);
			return suffix;
		}

		//
		suffix.SetLength(suffix.GetLength() + 1);

		// TODO: Also include roughness
		const bool postRecon = suffix.GetLength() > 0;

		// Decide if NEE or continue PT
		if (rng.NextFloat() < params.neeProb)
		{
			//
			suffix.p *= params.neeProb;

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
					const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
						interaction.meshSbtData->evalMaterialSbtIdx,
						interaction,
						lightDir);

					suffix.f *= brdfEvalResult.brdfResult * emitterSample.color;
					if (postRecon) { suffix.postReconF *= brdfEvalResult.brdfResult * emitterSample.color; }

					suffix.SetValid(true);
					validEmitterFound = true;
				}
			}

			if (!validEmitterFound) { suffix.SetValid(false); }
			return suffix;
		}

		// Indirect illumination, generate next ray
		const BrdfSampleResult brdfSampleResult = optixDirectCall<BrdfSampleResult, const Interaction&, PCG32&>(
			interaction.meshSbtData->sampleMaterialSbtIdx,
			interaction,
			rng);
		if (brdfSampleResult.samplingPdf <= 0.0f)
		{
			suffix.SetValid(false);
			return suffix;
		}

		// Store as reconnection vertex if fit
		if (postRecon && suffix.GetReconIdx() == 0)
		{
			suffix.reconInteraction = interaction;
			suffix.SetReconIdx(suffix.GetLength());
			suffix.reconOutDir = brdfSampleResult.outDir;
		}
		else if (postRecon)
		{
			suffix.postReconF *= brdfSampleResult.brdfVal;
		}
		
		currentPos = interaction.pos;
		currentDir = brdfSampleResult.outDir;

		suffix.f *= brdfSampleResult.brdfVal;
		suffix.p *= brdfSampleResult.samplingPdf * (1.0f - params.neeProb);
	}

	return suffix;
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
		Interaction interaction{};
		TraceWithDataPointer<Interaction>(
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
			//
			throughput /= params.neeProb;
			
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
					const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
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
		const BrdfSampleResult brdfSampleResult = optixDirectCall<BrdfSampleResult, const Interaction&, PCG32&>(
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
		throughput /= (1.0f - params.neeProb);
	}

	return radiance;
}
