#pragma once

#include <graph/Interaction.h>
#include <graph/LaunchParams.h>
#include <graph/restir/PrefixPath.h>
#include <graph/restir/SuffixPath.h>
#include <graph/restir/Reconnection.h>
#include <graph/sample_emitter.h>
#include <optix_device.h>
#include <graph/trace.h>

static constexpr uint32_t WINDOW_RADIUS = 10;
static constexpr uint32_t WINDOW_SIZE = 2 * WINDOW_RADIUS + 1;

static constexpr __forceinline__ __device__ glm::uvec2 SelectSpatialNeighbor(const glm::uvec2& pixelCoord, PCG32& rng)
{
	const uint32_t xCoord = (pixelCoord.x - WINDOW_RADIUS) + (rng.NextUint32() % WINDOW_SIZE);
	const uint32_t yCoord = (pixelCoord.y - WINDOW_RADIUS) + (rng.NextUint32() % WINDOW_SIZE);
	return glm::uvec2(xCoord, yCoord);
}

static constexpr __forceinline__ __device__ float CalcReconnectionJacobian(
	const glm::vec3& oldXi, 
	const glm::vec3& newXi, 
	const glm::vec3& targetPos, 
	const glm::vec3& targetNormal)
{
	const float term1 = glm::abs(glm::dot(glm::normalize(newXi - targetPos), targetNormal) / glm::dot(glm::normalize(oldXi - targetPos), targetNormal));
	const float term2 = glm::dot(targetPos - oldXi, targetPos - oldXi);
	const float term3 = glm::dot(targetPos - newXi, targetPos - newXi);
	const float result = term1 * term2 / term3;
	if (result > 2.0f || glm::isinf(result) || glm::isnan(result)) { return 0.0f; }
	return result;
}

static __forceinline__ __device__ PrefixPath TracePrefix(
	PrefixPath& prefix,
	const glm::vec3& origin,
	const glm::vec3& dir,
	const size_t maxLen,
	PCG32& rng,
	const LaunchParams& params)
{
	glm::vec3 currentPos = origin;
	glm::vec3 currentDir = dir;

	prefix.Reset();
	prefix.rng = rng;
	prefix.p = 1.0f;
	prefix.f = glm::vec3(1.0f);
	prefix.postReconF = glm::vec3(1.0f);
	prefix.SetValid(true);

	Interaction interaction{};

	// Trace
	bool postRecon = false;
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
			interaction);

		// Exit if no surface found
		if (!interaction.valid)
		{
			prefix.SetValid(false);
			break;
		}

		// Add to path length
		prefix.pathLen += glm::distance(currentPos, interaction.pos);

		// Inc vertex count
		prefix.SetLength(prefix.GetLength() + 1);
		
		// Store primary interaction
		if (prefix.GetLength() == 1)
		{
			prefix.primaryInt = interaction;
		}

		// Decide if NEE or continue PT
		const bool forceNee = params.rendererType == RendererType::RestirPt && traceIdx == maxLen - 1;
		if (rng.NextFloat() < params.neeProb || forceNee)
		{
			//
			prefix.p *= params.neeProb;

			// NEE
			bool validEmitterFound = false;
			size_t neeCounter = 0;
			while (!validEmitterFound && neeCounter < params.neeTries)
			{
				//
				++neeCounter;

				// Sample light source
				const EmitterSample emitterSample = SampleEmitter(rng, params.emitterTable);
				const glm::vec3 lightDir = glm::normalize(emitterSample.pos - interaction.pos);
				const float distance = glm::length(emitterSample.pos - interaction.pos);

				// Cast shadow ray
				const bool occluded = TraceOcclusion(
					interaction.pos,
					lightDir,
					1e-3f,
					distance,
					params);

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

		// Indirect illumination, generate next ray
		const BrdfSampleResult brdfSampleResult = optixDirectCall<BrdfSampleResult, const Interaction&, PCG32&>(
			interaction.meshSbtData->sampleMaterialSbtIdx,
			interaction,
			rng);
		if (brdfSampleResult.samplingPdf <= 0.0f)
		{
			prefix.SetValid(false);
			break;
		}

		// Check if this interaction can be a reconnection interaction
		const bool intCanRecon = 
			glm::distance(currentPos, interaction.pos) > params.restir.reconMinDistance && 
			brdfSampleResult.roughness > params.restir.reconMinRoughness;

		// Store as reconnection vertex if fit
		if (!postRecon && intCanRecon)
		{
			prefix.reconInt = interaction;
			prefix.SetReconIdx(prefix.GetLength());
			prefix.reconOutDir = brdfSampleResult.outDir;
			postRecon = true;
		}

		currentPos = interaction.pos;
		currentDir = brdfSampleResult.outDir;

		prefix.f *= brdfSampleResult.brdfVal;
		prefix.p *= brdfSampleResult.samplingPdf * (1.0f - params.neeProb);
		if (postRecon) { prefix.postReconF *= brdfSampleResult.brdfVal; }
	}

	prefix.lastInt = interaction;
	return prefix;
}

static __forceinline__ __device__ SuffixPath TraceSuffix(
	SuffixPath& suffix,
	const PrefixPath& prefix,
	PCG32& rng,
	const LaunchParams& params)
{
	suffix.Reset();
	suffix.f = glm::vec3(1.0f);
	suffix.p = 1.0f;
	suffix.postReconF = glm::vec3(1.0f);
	suffix.rng = rng;
	suffix.lastPrefixInt = prefix.lastInt;

	Interaction interaction{};

	// Calc max len
	const uint32_t maxLen = params.maxPathLen - prefix.GetLength();

	// Get last prefix interaction
	const Interaction lastPrefixInt(prefix.lastInt, params.transforms);
	if (!lastPrefixInt.valid)
	{
		suffix.SetValid(false);
		return suffix; 
	}

	// Suffix may directly terminate by NEE
	if (rng.NextFloat() < params.neeProb)
	{
		//
		suffix.p *= params.neeProb;

		// NEE
		bool validEmitterFound = false;
		size_t neeCounter = 0;
		while (!validEmitterFound && neeCounter < params.neeTries)
		{
			//
			++neeCounter;

			// Sample light source
			const EmitterSample emitterSample = SampleEmitter(rng, params.emitterTable);
			const glm::vec3 lightDir = glm::normalize(emitterSample.pos - lastPrefixInt.pos);
			const float distance = glm::length(emitterSample.pos - lastPrefixInt.pos);

			// Cast shadow ray
			const bool occluded = TraceOcclusion(
				lastPrefixInt.pos,
				lightDir,
				1e-3f,
				distance,
				params);

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
					lastPrefixInt.pos,
					lightDir,
					1e-3f,
					distance + 1.0f,
					params.surfaceTraceParams,
					interaction);
				if (!interaction.valid)
				{
					suffix.SetValid(false);
					return suffix;
				}
				suffix.reconInt = interaction;

				// Calc brdf
				const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
					lastPrefixInt.meshSbtData->evalMaterialSbtIdx,
					lastPrefixInt,
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
		lastPrefixInt.meshSbtData->sampleMaterialSbtIdx,
		lastPrefixInt,
		rng);
	if (brdfSampleResult.samplingPdf <= 0.0f)
	{
		suffix.SetValid(false);
		return suffix;
	}

	glm::vec3 currentPos = lastPrefixInt.pos;
	glm::vec3 currentDir = brdfSampleResult.outDir;
	suffix.f *= brdfSampleResult.brdfVal;
	suffix.p *= brdfSampleResult.samplingPdf * (1.0f - params.neeProb);

	// Trace
	bool postRecon = false;
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
			interaction);
		if (!interaction.valid)
		{
			suffix.SetValid(false);
			return suffix;
		}

		//
		suffix.SetLength(suffix.GetLength() + 1);

		// Decide if NEE or continue PT
		if (rng.NextFloat() < params.neeProb)
		{
			//
			suffix.p *= params.neeProb;

			//
			if (!postRecon)
			{
				suffix.reconInt = interaction;
				suffix.SetReconIdx(suffix.GetLength());
				suffix.reconOutDir = brdfSampleResult.outDir;
				postRecon = true;
			}

			// NEE
			bool validEmitterFound = false;
			size_t neeCounter = 0;
			while (!validEmitterFound && neeCounter < params.neeTries)
			{
				//
				++neeCounter;

				// Sample light source
				const EmitterSample emitterSample = SampleEmitter(rng, params.emitterTable);
				const glm::vec3 lightDir = glm::normalize(emitterSample.pos - interaction.pos);
				const float distance = glm::length(emitterSample.pos - interaction.pos);

				// Cast shadow ray
				const bool occluded = TraceOcclusion(
					interaction.pos,
					lightDir,
					1e-3f,
					distance,
					params);

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
					suffix.postReconF *= brdfEvalResult.brdfResult * emitterSample.color;

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

		// Check if this interaction can be a reconnection interaction
		const bool intCanRecon =
			glm::distance(currentPos, interaction.pos) > params.restir.reconMinDistance &&
			brdfSampleResult.roughness > params.restir.reconMinRoughness;

		// Store as reconnection vertex if fit
		if (!postRecon && intCanRecon)
		{
			suffix.reconInt = interaction;
			suffix.SetReconIdx(suffix.GetLength());
			suffix.reconOutDir = brdfSampleResult.outDir;
			postRecon = true;
		}
		
		currentPos = interaction.pos;
		currentDir = brdfSampleResult.outDir;

		suffix.f *= brdfSampleResult.brdfVal;
		suffix.p *= brdfSampleResult.samplingPdf * (1.0f - params.neeProb);
		if (postRecon) { suffix.postReconF *= brdfSampleResult.brdfVal; }
	}

	return suffix;
}

static __forceinline__ __device__ glm::vec3 TraceCompletePath(
	const glm::vec3& origin,
	const glm::vec3& dir,
	PCG32& rng,
	const LaunchParams& params)
{
	glm::vec3 currentPos = origin;
	glm::vec3 currentDir = dir;
	size_t currentDepth = 0;

	glm::vec3 radiance(0.0f);
	glm::vec3 throughput(1.0f);

	// Trace
	for (uint32_t traceIdx = 0; traceIdx < params.maxPathLen; ++traceIdx)
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
			interaction);

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
			while (!validEmitterFound && neeCounter < params.neeTries)
			{
				//
				++neeCounter;

				// Sample light source
				const EmitterSample emitterSample = SampleEmitter(rng, params.emitterTable);
				const glm::vec3 lightDir = glm::normalize(emitterSample.pos - interaction.pos);
				const float distance = glm::length(emitterSample.pos - interaction.pos);

				// Cast shadow ray
				const bool occluded = TraceOcclusion(
					interaction.pos,
					lightDir,
					1e-3f,
					distance,
					params);

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

static __forceinline__ __device__ bool TracePrefixForFinalGather(
	glm::vec3& throughput,
	float& p,
	Interaction& lastInt,
	const glm::vec3& origin,
	const glm::vec3& dir,
	const size_t maxLen,
	PCG32& rng,
	const LaunchParams& params)
{
	glm::vec3 currentPos = origin;
	glm::vec3 currentDir = dir;
	size_t currentDepth = 0;

	throughput = glm::vec3(1.0f);
	p = 1.0f;
	lastInt.valid = false;

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
			lastInt);

		// Exit if no surface found
		if (!lastInt.valid)
		{
			return false;
		}

		//
		++currentDepth;

		// Indirect illumination, generate next ray
		const BrdfSampleResult brdfSampleResult = optixDirectCall<BrdfSampleResult, const Interaction&, PCG32&>(
			lastInt.meshSbtData->sampleMaterialSbtIdx,
			lastInt,
			rng);
		if (brdfSampleResult.samplingPdf <= 0.0f)
		{
			return false;
		}

		currentPos = lastInt.pos;
		currentDir = brdfSampleResult.outDir;
		throughput *= brdfSampleResult.brdfVal;
		p *= brdfSampleResult.samplingPdf;
	}

	return true;
}
