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
#include <graph/restir/suffix_reuse.h>

__constant__ LaunchParams params;

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

	if (prefixSearchPayload.neighCount == 0) { return glm::vec3(0.0f); }
	if (!params.restir.showPrefixEntryContrib) { return glm::vec3(0.0f, 1.0f, 0.0f); }

	// Sum suffix contrib
	glm::vec3 contrib(0.0f);
	const uint32_t k = params.restir.gatherM - 1;
	const uint32_t offset = k * pixelIdx;
	for (uint32_t suffixIdx = 0; suffixIdx < prefixSearchPayload.neighCount; ++suffixIdx)
	{
		const uint32_t neighPixelIdx = params.restir.prefixNeighbors[offset + suffixIdx].pixelIdx;
		const SuffixPath& suffix = params.restir.suffixReservoirs[2 * neighPixelIdx + params.restir.frontBufferIdx].sample;
		contrib += suffix.f;
	}
	return contrib / static_cast<float>(prefixSearchPayload.neighCount);
}

static __forceinline__ __device__ glm::vec3 FinalGatherCanon(
	const uint32_t pixelIdx,
	const glm::vec3& origin,
	const glm::vec3& dir,
	const uint32_t k,
	PCG32& rng)
{
	// Trace canon prefix
	PrefixPath canonPrefix{};
	TracePrefix(canonPrefix, origin, dir, rng, params);
	if (!canonPrefix.IsValid()) { return glm::vec3(0.0f); }

	// Exit if canon has nee
	if (canonPrefix.IsNee()) { return canonPrefix.f / canonPrefix.p; }

	// Trace canon suffix
	SuffixPath canonSuffix{};
	TraceSuffix(canonSuffix, canonPrefix, rng, params);
	if (!canonSuffix.IsValid()) { return glm::vec3(0.0f); }
	const glm::vec3 canonContrib = canonPrefix.f * canonSuffix.f / (canonPrefix.p * canonSuffix.p);

	// Exit if k = 0
	if (k == 0) { return canonContrib; }

	// Get last prefix interaction
	const Interaction lastPrefixInt(canonSuffix.lastPrefixInt, params.transforms);

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

	// Track prefix stats
	if (params.restir.trackPrefixStats)
	{
		atomicMin(&params.restir.prefixStats[0].minNeighCount, neighCount);
		atomicMax(&params.restir.prefixStats[0].maxNeighCount, neighCount);
		atomicAdd(&params.restir.prefixStats[0].totalNeighCount, neighCount);
	}

	// Count valid neighbors
	uint32_t validNeighCount = 0;
	uint32_t validNeighFlags = 0;
	for (uint32_t neighIdx = 0; neighIdx < neighCount; ++neighIdx)
	{
		// Get neighbor pixel index
		const uint32_t neighPixelIdx = prefixSearchPayload.GetNeighbor(neighIdx, params).pixelIdx;

		// Check if neigh suffix is valid
		const Reservoir<SuffixPath>& neighSuffixRes = params.restir.suffixReservoirs[2 * neighPixelIdx + params.restir.frontBufferIdx];
		const SuffixPath& neighSuffix = neighSuffixRes.sample;
		if (!neighSuffix.IsValid()) { continue; }

		// Mark as valid
		++validNeighCount;
		validNeighFlags |= 1 << neighIdx;
	}

	// Borrow their suffixes and gather path contributions
	glm::vec3 suffixContrib(0.0f);
	const float validNeighCountF = static_cast<float>(validNeighCount);
	const float pairwiseK = validNeighCountF;
	float canonMisWeight = 1.0f;

	for (size_t neighIdx = 0; neighIdx < neighCount; ++neighIdx)
	{
		// Check if suffix is valid
		if (!(validNeighFlags & (1 << neighIdx))) { continue; }

		// Get suffix
		const uint32_t neighPixelIdx = prefixSearchPayload.GetNeighbor(neighIdx, params).pixelIdx;
		const Reservoir<SuffixPath>& neighSuffixRes = params.restir.suffixReservoirs[2 * neighPixelIdx + params.restir.frontBufferIdx];
		const SuffixPath& neighSuffix = neighSuffixRes.sample;

		// Shift
		float jacobianNeighToCanon = 0.0f;
		const glm::vec3 fFromCanonOfNeigh = CalcCurrContribInOtherDomain(neighSuffix, canonSuffix, jacobianNeighToCanon, params);
		const float pFromCanonOfNeigh = GetLuminance(fFromCanonOfNeigh) * jacobianNeighToCanon;

		float jacobianCanonToNeigh = 0.0f;
		const glm::vec3 fFromNeighOfCanon = CalcCurrContribInOtherDomain(canonSuffix, neighSuffix, jacobianCanonToNeigh, params);
		const float pFromNeighOfCanon = GetLuminance(fFromNeighOfCanon) * jacobianCanonToNeigh;

		const glm::vec3& fFromCanonOfCanon = canonSuffix.f;
		const glm::vec3& fFromNeighOfNeigh = neighSuffix.f;
		const float pFromCanonOfCanon = GetLuminance(fFromCanonOfCanon);
		const float pFromNeighOfNeigh = GetLuminance(fFromNeighOfNeigh);

		// Calc neigh mis weight
		float neighMisWeight = ComputeNeighborPairwiseMISWeight(
			fFromCanonOfNeigh, fFromNeighOfNeigh, jacobianNeighToCanon, pairwiseK, 1.0f, neighSuffixRes.confidence);
		if (glm::isnan(neighMisWeight) || glm::isinf(neighMisWeight)) neighMisWeight = 0.0f;

		// Add to suffix contribution
		suffixContrib += neighMisWeight * fFromCanonOfNeigh * neighSuffixRes.wSum * jacobianNeighToCanon;
		//suffixContrib += neighMisWeight * fFromNeighOfNeigh * neighSuffixRes.wSum;

		// Update canon mis weight
		canonMisWeight += ComputeCanonicalPairwiseMISWeight(
			fFromCanonOfCanon, fFromNeighOfCanon, jacobianCanonToNeigh, pairwiseK, 1.0f, neighSuffixRes.confidence);
	}

	// Gather
	glm::vec3 gatherContrib = suffixContrib * canonPrefix.f / canonPrefix.p;
	gatherContrib /= static_cast<float>(validNeighCount + 1); // 1 / N
	
	glm::vec3 result = gatherContrib + (canonContrib * canonMisWeight);
	result /= pairwiseK + 1.0f; // 1 / (k + 1) for pairwise MIS

	if (glm::any(glm::isinf(result) || glm::isnan(result))) { return glm::vec3(0.0f); }
	return result;
}

static __forceinline__ __device__ glm::vec3 FinalGatherNotCanon(
	const uint32_t pixelIdx,
	const glm::vec3& origin, 
	const glm::vec3& dir,
	const uint32_t k,
	PCG32& rng)
{
	printf("Hi\n");

	// Exit if k = 0
	if (k == 0) { return glm::vec3(0.0f); }

	// Trace new prefix for pixel q
	Interaction lastPrefixInt{};
	glm::vec3 prefixThroughput(0.0f);
	float prefixP = 0.0f;
	if (!TracePrefixForFinalGather(prefixThroughput, prefixP, lastPrefixInt, origin, dir, params.restir.prefixLen, rng, params))
	{
		return glm::vec3(0.0f);
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
	
	// Track prefix stats
	if (params.restir.trackPrefixStats)
	{
		atomicMin(&params.restir.prefixStats[0].minNeighCount, neighCount);
		atomicMax(&params.restir.prefixStats[0].maxNeighCount, neighCount);
		atomicAdd(&params.restir.prefixStats[0].totalNeighCount, neighCount);
	}

	// Count valid neighbors
	uint32_t validNeighCount = 0;
	uint32_t validNeighFlags = 0;
	for (uint32_t neighIdx = 0; neighIdx < neighCount; ++neighIdx)
	{
		// Get neighbor pixel index
		const uint32_t neighPixelIdx = prefixSearchPayload.GetNeighbor(neighIdx, params).pixelIdx;

		// Check if neigh suffix is valid
		const Reservoir<SuffixPath>& neighSuffixRes = params.restir.suffixReservoirs[2 * neighPixelIdx + params.restir.frontBufferIdx];
		const SuffixPath& neighSuffix = neighSuffixRes.sample;
		if (!neighSuffix.IsValid()) { continue; }

		// Mark as valid
		++validNeighCount;
		validNeighFlags |= 1 << neighIdx;
	}

	// Borrow their suffixes and gather path contributions
	glm::vec3 suffixContrib(0.0f);
	const float validNeighCountF = static_cast<float>(validNeighCount);
	const float pairwiseK = validNeighCountF;

	for (size_t neighIdx = 0; neighIdx < neighCount; ++neighIdx)
	{
		// Check if suffix is valid
		if (!(validNeighFlags & (1 << neighIdx))) { continue; }

		// Get suffix
		const uint32_t neighPixelIdx = prefixSearchPayload.GetNeighbor(neighIdx, params).pixelIdx;
		const Reservoir<SuffixPath>& neighSuffixRes = params.restir.suffixReservoirs[2 * neighPixelIdx + params.restir.frontBufferIdx];
		const SuffixPath& neighSuffix = neighSuffixRes.sample;

		// Shift
		float jacobianNeighToCanon = 0.0f;
		const glm::vec3 fFromCanonOfNeigh = CalcCurrContribInOtherDomain(neighSuffix, lastPrefixInt, jacobianNeighToCanon, params);
		const float pFromCanonOfNeigh = GetLuminance(fFromCanonOfNeigh) * jacobianNeighToCanon;

		const glm::vec3& fFromNeighOfNeigh = neighSuffix.f;
		const float pFromNeighOfNeigh = GetLuminance(fFromNeighOfNeigh);

		// Calc neigh mis weight
		float neighMisWeight = ComputeNeighborPairwiseMISWeight(
			fFromCanonOfNeigh, fFromNeighOfNeigh, jacobianNeighToCanon, pairwiseK, 1.0f, neighSuffixRes.confidence);
		if (glm::isnan(neighMisWeight) || glm::isinf(neighMisWeight)) neighMisWeight = 0.0f;

		// Add to suffix contribution
		suffixContrib += neighMisWeight * fFromCanonOfNeigh * neighSuffixRes.wSum * jacobianNeighToCanon;
	}

	// Gather
	glm::vec3 result = suffixContrib * prefixThroughput / prefixP;
	result /= static_cast<float>(params.restir.gatherN);
	result /= pairwiseK + 1.0f;

	if (glm::any(glm::isinf(result) || glm::isnan(result))) { return glm::vec3(0.0f); }
	return result;
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

	// Final gather
	const size_t k = params.restir.gatherM - 1;
	outputRadiance += FinalGatherCanon(pixelIdx, origin, dir, k, rng);
	for (size_t prefixIdx = 1; prefixIdx < params.restir.gatherN; ++prefixIdx)
	{
		outputRadiance += FinalGatherNotCanon(pixelIdx, origin, dir, k, rng);
	}

	// Check output radiance
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
