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

static constexpr __forceinline__ __device__ float ComputeCanonicalPairwiseMISWeight(
	const glm::vec3& basisPathContributionAtBasis, 
	const glm::vec3& basisPathContributionAtNeighbor, 
	const float basisPathToNeighborJacobian,
	const float pairwiseK, 
	const float canonicalM, 
	const float neighborM)
{
	float misWeightBasisPath = 1.f;

	if (GetLuminance(basisPathContributionAtBasis) > 0.f)
	{
		float atBasisTerm = GetLuminance(basisPathContributionAtBasis) * canonicalM;
		misWeightBasisPath = atBasisTerm / (atBasisTerm + GetLuminance(basisPathContributionAtNeighbor) * basisPathToNeighborJacobian * neighborM * pairwiseK);
	}
	return misWeightBasisPath;
}

static constexpr __forceinline__ __device__ __host__ float ComputeNeighborPairwiseMISWeight(
	const glm::vec3& neighborPathContributionAtBasis, 
	const glm::vec3& neighborPathContributionAtNeighbor,
	const float neighborPathToBasisJacobian, 
	const float pairwiseK, 
	const float canonicalM, 
	const float neighborM)
{
	float misWeightNeighborPath = 0.f;
	if (GetLuminance(neighborPathContributionAtNeighbor) > 0.f)
	{
		misWeightNeighborPath = GetLuminance(neighborPathContributionAtNeighbor) * neighborM /
			(GetLuminance(neighborPathContributionAtNeighbor) * neighborM + GetLuminance(neighborPathContributionAtBasis) * neighborPathToBasisJacobian * canonicalM / pairwiseK);
	}
	return misWeightNeighborPath;
}

static constexpr __forceinline__ __device__ glm::uvec2 SelectSpatialNeighbor(const glm::uvec2& pixelCoord, PCG32& rng)
{
	const uint32_t xCoord = (pixelCoord.x + WINDOW_RADIUS) - (rng.NextUint32() % WINDOW_SIZE);
	const uint32_t yCoord = (pixelCoord.y + WINDOW_RADIUS) - (rng.NextUint32() % WINDOW_SIZE);
	const glm::uvec2 result(xCoord, yCoord);
	if (result == pixelCoord) return pixelCoord + glm::uvec2(1);
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
	if (glm::isinf(result) || glm::isnan(result)) { return 0.0f; }
	//if (result > 4.0f) { return 0.0f; }
	return glm::max(0.0f, result);
}

static __forceinline__ __device__ void TracePrefix(
	PrefixPath& prefix,
	const glm::vec3& origin,
	const glm::vec3& dir,
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

	//
	const uint32_t maxLen = params.rendererType == RendererType::RestirPt ? params.maxPathLen : params.restir.prefixLen;

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
			return;
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

		// 
		if (params.rendererType == RendererType::ConditionalRestir && traceIdx == maxLen - 1)
		{
			if (!postRecon)
			{
				prefix.reconInt = interaction;
				prefix.SetReconIdx(prefix.GetLength());
				prefix.reconOutDir = glm::vec3(0.0f);
			}

			prefix.SetValid(true);
			prefix.lastInt = interaction;
			return;
		}

		// Decide if NEE or continue PT
		const bool forceNee = params.rendererType == RendererType::RestirPt && traceIdx == maxLen - 1;
		if (rng.NextFloat() < params.neeProb || forceNee)
		{
			//
			prefix.p *= params.neeProb;

			// NEE
			// Sample light source
			const EmitterSample emitterSample = SampleEmitter(rng, params.emitterTable);
			const glm::vec3 lightDir = glm::normalize(emitterSample.pos - interaction.pos);
			const float distance = glm::length(emitterSample.pos - interaction.pos);

			// Cast shadow ray
			if (TraceOcclusion(interaction.pos, lightDir, 1e-3f, distance, params))
			{
				prefix.SetValid(false);
				return;
			}

			// Create interaction at light source
			if (!postRecon)
			{
				TraceWithDataPointer<Interaction>(
					params.traversableHandle,
					currentPos,
					currentDir,
					1e-3f,
					1e16f,
					params.surfaceTraceParams,
					interaction);
				if (interaction.valid)
				{
					prefix.reconInt = interaction;
					prefix.SetReconIdx(prefix.GetLength());
					prefix.reconOutDir = currentDir;
					postRecon = true;
				}
				else
				{
					prefix.SetValid(false);
					return;
				}
			}

			// Calc brdf
			const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
				interaction.meshSbtData->evalMaterialSbtIdx,
				interaction,
				lightDir);
			if (brdfEvalResult.samplingPdf <= 0.0f) 
			{
				prefix.SetValid(false);
				return;
			}

			const float distFactor = 1.0f / (distance * distance);

			prefix.f *= brdfEvalResult.brdfResult * emitterSample.color * distFactor;
			if (postRecon) { prefix.postReconF *= brdfEvalResult.brdfResult * emitterSample.color * distFactor; }
			if (prefix.GetLength() == 1) { prefix.f += brdfEvalResult.emission * distFactor; }

			prefix.SetNee(true);
			prefix.SetValid(true);
			prefix.lastInt = interaction;

			return;
		}

		// Indirect illumination, generate next ray
		const BrdfSampleResult brdfSampleResult = optixDirectCall<BrdfSampleResult, const Interaction&, PCG32&>(
			interaction.meshSbtData->sampleMaterialSbtIdx,
			interaction,
			rng);
		if (brdfSampleResult.samplingPdf <= 0.0f)
		{
			prefix.SetValid(false);
			return;
		}

		// Check if this interaction can be a reconnection interaction
		const float reconDistance = glm::distance(currentPos, interaction.pos);
		const float reconRoughness = brdfSampleResult.roughness;
		const bool intCanRecon =
			reconDistance > params.restir.reconMinDistance &&
			reconRoughness > params.restir.reconMinRoughness &&
			prefix.GetLength() >= 2;

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
}

static __forceinline__ __device__ void TraceSuffix(
	SuffixPath& suffix,
	const PrefixPath& prefix,
	PCG32& rng,
	const LaunchParams& params)
{
	// Init suffix
	suffix.f = glm::vec3(1.0f);
	suffix.p = 1.0f;
	suffix.postReconF = glm::vec3(1.0f);
	suffix.rng = rng;
	suffix.lastPrefixInt = prefix.lastInt;
	suffix.Reset();
	
	// Check prefix
	if (!prefix.IsValid() || prefix.IsNee()) { return; }

	// Calc max len
	const uint32_t maxLen = params.maxPathLen - prefix.GetLength();

	// Get last prefix interaction
	Interaction interaction(prefix.lastInt, params.transforms);
	if (!interaction.valid) { return; }
	
	//
	bool postRecon = false;
	glm::vec3 lastDir(0.0f);
	for (uint32_t vertIdx = 0; vertIdx < maxLen; ++vertIdx)
	{
		// Trace new interaction
		const glm::vec3 lastPos = interaction.pos;
		if (suffix.GetLength() > 0)
		{
			TraceWithDataPointer<Interaction>(
				params.traversableHandle,
				lastPos,
				lastDir,
				1e-3f,
				1e16f,
				params.surfaceTraceParams,
				interaction);
			if (!interaction.valid) { return; }
		}

		// Nee
		const bool forceNee = vertIdx == maxLen - 1;
		if (rng.NextFloat() < params.neeProb || forceNee)
		{
			// Prob
			if (!forceNee) { suffix.p *= params.neeProb; }

			// Sample light source
			const EmitterSample emitter = SampleEmitter(rng, params.emitterTable);

			// Check occlusion
			const glm::vec3 lightVec = emitter.pos - interaction.pos;
			const float lightLen = glm::length(lightVec);
			const glm::vec3 lightDir = glm::normalize(lightVec);
			if (glm::any(glm::isnan(lightDir) || glm::isinf(lightDir))) { return; }
			if (TraceOcclusion(interaction.pos, lightDir, 1e-3f, lightLen, params)) { return; }

			// Eval brdf
			const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
				interaction.meshSbtData->evalMaterialSbtIdx,
				interaction,
				lightDir);
			if (brdfEvalResult.samplingPdf <= 0.0f) { return; }

			// Increment length
			suffix.SetLength(suffix.GetLength() + 1);

			// Create reconnection interaction if not already done
			if (!postRecon)
			{
				Interaction reconInt{};
				TraceWithDataPointer<Interaction>(
					params.traversableHandle,
					interaction.pos,
					lightDir,
					1e-3f,
					1e16f,
					params.surfaceTraceParams,
					reconInt);
				if (!reconInt.valid) { return; }

				suffix.reconInt = reconInt;
				suffix.SetReconIdx(suffix.GetLength());
				suffix.reconOutDir = glm::vec3(0.0f);
				postRecon = true;
			}

			// Radiance
			const float distFactor = 1.0f / (lightLen * lightLen);
			suffix.f *= brdfEvalResult.brdfResult * emitter.color * distFactor;
			suffix.postReconF *= emitter.color * distFactor;
			if (suffix.GetReconIdx() < suffix.GetLength()) { suffix.postReconF *= brdfEvalResult.brdfResult; }

			// End
			suffix.SetValid(true);
			return;
		}

		// Sample brdf
		const BrdfSampleResult brdfSampleResult = optixDirectCall<BrdfSampleResult, const Interaction&, PCG32&>(
			interaction.meshSbtData->sampleMaterialSbtIdx,
			interaction,
			rng);
		if (brdfSampleResult.samplingPdf <= 0.0f) { return; }

		// Update f and p
		suffix.p *= brdfSampleResult.samplingPdf * (1.0f - params.neeProb);
		suffix.f *= brdfSampleResult.brdfVal;
		if (postRecon) { suffix.postReconF *= brdfSampleResult.brdfVal; }

		// Check if this interaction can be a reconnection interaction
		const float reconDistance = glm::distance(lastPos, interaction.pos);
		const float reconRoughness = brdfSampleResult.roughness;
		const bool intCanRecon =
			reconDistance > params.restir.reconMinDistance &&
			reconRoughness > params.restir.reconMinRoughness &&
			prefix.GetLength() >= 1;

		// Store as reconnection vertex if fit
		if (!postRecon && intCanRecon)
		{
			suffix.reconInt = interaction;
			suffix.SetReconIdx(prefix.GetLength());
			suffix.reconOutDir = brdfSampleResult.outDir;
			postRecon = true;
		}

		// Update
		lastDir = brdfSampleResult.outDir;
		suffix.SetLength(suffix.GetLength() + 1);
	}
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
			// Sample light source
			const EmitterSample emitterSample = SampleEmitter(rng, params.emitterTable);
			const glm::vec3 lightDir = glm::normalize(emitterSample.pos - interaction.pos);
			const float distance = glm::length(emitterSample.pos - interaction.pos);

			// Cast shadow ray
			if (TraceOcclusion(interaction.pos, lightDir, 1e-3f, distance, params)) { return glm::vec3(0.0f); }

			// If emitter is not occluded -> end NEE
			// Calc brdf
			const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
				interaction.meshSbtData->evalMaterialSbtIdx,
				interaction,
				lightDir);

			// Radiance
			const float distFactor = 1.0f / (distance * distance);
			radiance = throughput * brdfEvalResult.brdfResult * emitterSample.color * distFactor;
			if (currentDepth == 1) { radiance += brdfEvalResult.emission; }

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
