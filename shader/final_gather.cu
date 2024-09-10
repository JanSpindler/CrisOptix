#include <cuda_runtime.h>
#include <optix_device.h>
#include <graph/LaunchParams.h>
#include <util/glm_cuda.h>
#include <util/pixel_index.h>
#include <graph/trace.h>
#include <graph/restir/PrefixSearchPayload.h>
#include <graph/restir/path_gen.h>
#include <cuda/std/tuple>
#include <device_atomic_functions.hpp>

__constant__ LaunchParams params;

static __forceinline__ __device__ cuda::std::pair<glm::vec3, float> ShiftSuffix(
	const Interaction& prefixLastInt,
	const SuffixPath& suffix,
	const float suffixUcwSrcDomain)
{
	// Get last prefix interaction
	if (!prefixLastInt.valid) { return { glm::vec3(0.0f), 0.0f }; }

	// Get suffix reconnection interaction
	Interaction reconInt(suffix.reconInt, params.transforms);
	if (!reconInt.valid) { return { glm::vec3(0.0f), 0.0f }; }

	// Hybrid shift
	const uint32_t reconVertCount = glm::max<int>(suffix.GetReconIdx() - 1, 0);
	Interaction currInt = prefixLastInt;
	PCG32 otherRng = suffix.rng;
	glm::vec3 throughput(1.0f);
	for (uint32_t idx = 0; idx < reconVertCount; ++idx)
	{
		//printf("%d\n", otherSuffix.GetReconIdx());

		// Sampled brdf
		const BrdfSampleResult brdf = optixDirectCall<BrdfSampleResult, const Interaction&, PCG32&>(
			currInt.meshSbtData->sampleMaterialSbtIdx,
			currInt,
			otherRng);
		if (brdf.samplingPdf <= 0.0f) { return { glm::vec3(0.0f), 0.0f }; }
		throughput *= brdf.brdfVal;

		// Recon dir
		const glm::vec3& reconDir = brdf.outDir;

		// Trace new interaction
		const glm::vec3 oldPos = currInt.pos;
		TraceWithDataPointer<Interaction>(params.traversableHandle, oldPos, reconDir, 1e-3f, 1e16f, params.surfaceTraceParams, currInt);
		if (!currInt.valid) { return { glm::vec3(0.0f), 0.0f }; }

		// Trace occlusion
		if (TraceOcclusion(oldPos, reconDir, 1e-3f, glm::distance(oldPos, currInt.pos), params)) { return { glm::vec3(0.0f), 0.0f }; }
	}

	// Trace occlusion
	const glm::vec3 reconVec = reconInt.pos - currInt.pos;
	const float reconLen = glm::length(reconVec);
	const glm::vec3 reconDir = glm::normalize(reconVec);

	if (glm::any(glm::isinf(reconDir) || glm::isnan(reconDir))) { return { glm::vec3(0.0f), 0.0f }; }
	if (TraceOcclusion(currInt.pos, reconDir, 1e-3f, reconLen, params)) { return { glm::vec3(0.0f), 0.0f }; }

	// Eval brdf at last prefix vert with new out dir
	const BrdfEvalResult brdfEvalResult1 = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
		currInt.meshSbtData->evalMaterialSbtIdx,
		currInt,
		reconDir);
	if (brdfEvalResult1.samplingPdf <= 0.0f) { return { glm::vec3(0.0f), 0.0f }; }
	throughput *= brdfEvalResult1.brdfResult;

	// 
	reconInt.inRayDir = reconDir;

	//
	if (suffix.GetReconIdx() > 0)
	{
		// Eval brdf at suffix recon vertex
		const BrdfEvalResult brdfEvalResult2 = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
			reconInt.meshSbtData->evalMaterialSbtIdx,
			reconInt,
			suffix.reconOutDir);
		if (brdfEvalResult2.samplingPdf <= 0.0f) { return { glm::vec3(0.0f), 0.0f }; }
		throughput *= brdfEvalResult2.brdfResult;
	}

	// Calc total contribution of new path
	const glm::vec3 radiance = throughput * suffix.postReconF;

	// Get suffix last prefix int
	const Interaction suffixLastPrefixInt(suffix.lastPrefixInt, params.transforms);
	if (!suffixLastPrefixInt.valid) { return { glm::vec3(0.0f), 0.0f }; }

	// Calc jacobian
	const float jacobian = CalcReconnectionJacobian(
		suffixLastPrefixInt.pos, 
		currInt.pos,
		reconInt.pos,
		reconInt.normal);

	// Return
	return { radiance, suffixUcwSrcDomain * jacobian };
}

static __forceinline__ __device__ glm::vec3 ShowPrefixEntries(const glm::uvec3& launchIdx, const size_t pixelIdx)
{
	// Init RNG
	PCG32& rng = params.restir.restirGBuffers[pixelIdx].rng;

	// Final gather
	// Spawn camera ray
	glm::vec3 origin(0.0f);
	glm::vec3 dir(0.0f);
	glm::vec2 uv = (glm::vec2(launchIdx) + rng.Next2d()) / glm::vec2(params.width, params.height);
	uv = 2.0f * uv - 1.0f; // [0, 1] -> [-1, 1]
	SpawnCameraRay(params.cameraData, uv, origin, dir);

	// Sample surface interaction
	Interaction interaction{};
	TraceWithDataPointer<Interaction>(
		params.traversableHandle,
		origin,
		dir,
		1e-3f,
		1e16f,
		params.surfaceTraceParams,
		interaction);
	if (!interaction.valid) { return glm::vec3(0.0f); }

	// Find k neighboring prefixes in world space
	static constexpr float EPSILON = 1e-16f;
	PrefixSearchPayload prefixSearchPayload(pixelIdx);
	TraceWithDataPointer<PrefixSearchPayload>(
		params.restir.prefixEntriesTraversHandle,
		interaction.pos,
		glm::vec3(EPSILON),
		0.0f,
		EPSILON,
		params.restir.prefixEntriesTraceParams,
		prefixSearchPayload);

	return prefixSearchPayload.neighCount ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(0.0f);
}

static __forceinline__ __device__ glm::vec3 GetRadiance(const glm::uvec3& launchIdx, const size_t pixelIdx)
{
	// Exit if primary interaction did not hit
	const PrefixPath& prefix = params.restir.canonicalPrefixes[pixelIdx];
	if (!prefix.primaryInt.IsValid()) { return glm::vec3(0.0f); }

	// Init RNG
	PCG32& rng = params.restir.restirGBuffers[pixelIdx].rng;

	// Init empty output radiance
	glm::vec3 outputRadiance(0.0f);

	// Final gather
	// Spawn camera ray
	glm::vec3 origin(0.0f);
	glm::vec3 dir(0.0f);
	glm::vec2 uv = (glm::vec2(launchIdx) + rng.Next2d()) / glm::vec2(params.width, params.height);
	uv = 2.0f * uv - 1.0f; // [0, 1] -> [-1, 1]
	SpawnCameraRay(params.cameraData, uv, origin, dir);

	// K = M - 1
	float canonSuffixMisWeight = 1.0f;
	const size_t k = params.restir.gatherM - 1;
	if (k > 0)
	{
		Interaction lastPrefixInt{};
		glm::vec3 prefixThroughput(0.0f);
		float prefixP = 0.0f;

		for (size_t prefixIdx = 0; prefixIdx < params.restir.gatherN; ++prefixIdx)
		{
			// Trace new prefix for pixel q
			if (!TracePrefixForFinalGather(prefixThroughput, prefixP, lastPrefixInt, origin, dir, params.restir.prefixLen, rng, params))
			{
				continue;
			}
			
			// Find k neighboring prefixes in world space
			static constexpr float EPSILON = 1e-16f;
			PrefixSearchPayload prefixSearchPayload(pixelIdx);
			TraceWithDataPointer<PrefixSearchPayload>(
				params.restir.prefixEntriesTraversHandle,
				lastPrefixInt.pos,
				glm::vec3(EPSILON),
				0.0f,
				EPSILON,
				params.restir.prefixEntriesTraceParams,
				prefixSearchPayload);
			const uint32_t neighCount = prefixSearchPayload.neighCount;
			const float misWeight = 1.0f / static_cast<float>(neighCount + 1.0f);

			// Set mis weight for canonical suffix
			if (prefixIdx == 0) { canonSuffixMisWeight = misWeight; }

			// Track prefix stats
			if (params.restir.trackPrefixStats)
			{
				atomicMin(&params.restir.prefixStats[0].minNeighCount, neighCount);
				atomicMax(&params.restir.prefixStats[0].maxNeighCount, neighCount);
				atomicAdd(&params.restir.prefixStats[0].totalNeighCount, neighCount);
			}

			// Borrow their suffixes and gather path contributions
			glm::vec3 suffixContrib(0.0f);
			for (size_t suffixIdx = 0; suffixIdx < neighCount; ++suffixIdx)
			{
				// Assume: Neighbor prefix and suffix are valid

				// Get suffix
				const uint32_t suffixPixelIdx = params.restir.prefixNeighbors[k * pixelIdx + suffixIdx].pixelIdx;
				const Reservoir<SuffixPath>& neighSuffixRes = params.restir.suffixReservoirs[2 * suffixPixelIdx + params.restir.frontBufferIdx];
				const SuffixPath& neighSuffix = neighSuffixRes.sample;

				// Shift suffix
				const cuda::std::pair<glm::vec3, float> shiftedSuffix = ShiftSuffix(
					lastPrefixInt, 
					neighSuffix, 
					neighSuffixRes.wSum / GetLuminance(neighSuffix.f));
				const glm::vec3& shiftedF = shiftedSuffix.first;
				const float ucwSuffix = shiftedSuffix.second;

				// Add
				suffixContrib += misWeight * shiftedF * ucwSuffix;
			}

			// Gather
			glm::vec3 result = suffixContrib * prefixThroughput / prefixP;
			if (glm::any(glm::isinf(result) || glm::isnan(result))) { result = glm::vec3(0.0f); }
			outputRadiance += result;
		}

		outputRadiance /= static_cast<float>(params.restir.gatherN);
	}

	// Add canon suffix contrib
	outputRadiance += canonSuffixMisWeight * TraceCompletePath(origin, dir, rng, params);

	if (glm::any(glm::isnan(outputRadiance) || glm::isinf(outputRadiance))) { outputRadiance = glm::vec3(0.0f); }

	return outputRadiance;
}

extern "C" __global__ void __raygen__final_gather()
{
	//
	const glm::uvec3 launchIdx = cuda2glm(optixGetLaunchIndex());
	const glm::uvec3 launchDims = cuda2glm(optixGetLaunchDimensions());
	const glm::uvec2 pixelCoord = glm::uvec2(launchIdx);
	
	// Exit if invalid launch idx
	if (launchIdx.x >= params.width || launchIdx.y >= params.height || launchIdx.z >= 1)
	{
		return;
	}

	// Show prefix entries
	const size_t pixelIdx = GetPixelIdx(pixelCoord, params);
	if (params.restir.showPrefixEntries)
	{
		params.outputBuffer[pixelIdx] = ShowPrefixEntries(launchIdx, pixelIdx);
		return;
	}

	// Accum
	const glm::vec3 outputRadiance = GetRadiance(launchIdx, pixelIdx);
	if (params.enableAccum)
	{
		const glm::vec3 oldVal = params.outputBuffer[pixelIdx];
		const float blendFactor = 1.0f / static_cast<float>(params.frameIdx + 1);
		params.outputBuffer[pixelIdx] = blendFactor * outputRadiance + (1.0f - blendFactor) * oldVal;
	}
	else
	{
		params.outputBuffer[pixelIdx] = outputRadiance;
	}
}
