#include <cuda_runtime.h>
#include <graph/LaunchParams.h>
#include <util/glm_cuda.h>
#include <util/pixel_index.h>
#include <graph/restir/suffix_reuse.h>

__constant__ LaunchParams params;

static __forceinline__ __device__ void SuffixSpatialReuse(const glm::uvec2& pixelCoord)
{
	// Get current prefix
	const size_t currPixelIdx = GetPixelIdx(pixelCoord, params);
	const PrefixPath& prefix = params.restir.prefixReservoirs[2 * currPixelIdx + params.restir.frontBufferIdx].sample;
	if (!prefix.IsValid() || prefix.IsNee()) { return; }

	// Get current suffix
	Reservoir<SuffixPath>& currRes = params.restir.suffixReservoirs[2 * currPixelIdx + params.restir.frontBufferIdx];
	const SuffixPath& currSuffix = currRes.sample;
	if (!currSuffix.IsValid()) { return; }

	// Get canonical suffix
	const SuffixPath& canonSuffix = params.restir.canonicalSuffixes[currPixelIdx];
	if (!canonSuffix.IsValid()) { return; }
	const float canonPHat = GetLuminance(canonSuffix.f);

	// Get rng
	PCG32& rng = params.restir.restirGBuffers[currPixelIdx].rng;

	// Cache data from current reservoir and reset
	const float currResWSum = currRes.wSum;
	const float currResConfidence = currRes.confidence;
	currRes.Reset();

	// RIS with pairwise MIS weights
	const uint32_t neighCount = params.restir.suffixSpatialCount;
	float canonMisWeight = 0.0f;
	
	for (uint32_t neighIdx = 0; neighIdx < neighCount; ++neighIdx)
	{
		// Select new neighbor
		const glm::uvec2 neighPixelCoord = SelectSpatialNeighbor(pixelCoord, rng);
		if (!IsPixelValid(neighPixelCoord, params)) { continue; }
		const uint32_t neighPixelIdx = GetPixelIdx(neighPixelCoord, params);

		// Get neighbor res and prefix
		const Reservoir<SuffixPath>& neighRes = params.restir.suffixReservoirs[2 * neighPixelIdx + params.restir.frontBufferIdx];
		const SuffixPath& neighSuffix = neighRes.sample;
		if (!neighSuffix.IsValid()) { continue; }

		// Get neighbor primary hit
		Interaction neighLastPrefixInt{};
		TraceInteractionSeed(neighSuffix.lastPrefixIntSeed, neighLastPrefixInt, params);
		if (!neighLastPrefixInt.valid) { continue; }

		// Shift
		float jacobianNeighToCanon = 0.0f;
		const glm::vec3 fFromCanonOfNeigh = CalcCurrContribInOtherDomain(neighSuffix, canonSuffix, jacobianNeighToCanon, params);
		const float pFromCanonOfNeigh = GetLuminance(fFromCanonOfNeigh) * jacobianNeighToCanon;

		float jacobianCanonToNeigh = 0.0f;
		const glm::vec3 fFromNeighOfCanon = CalcCurrContribInOtherDomain(canonSuffix, neighSuffix, jacobianCanonToNeigh, params);
		const float pFromNeighOfCanon = GetLuminance(fFromNeighOfCanon) * jacobianCanonToNeigh;

		// Calc neigh mis weight
		const float neighMisWeight = neighRes.confidence * GetLuminance(neighSuffix.f) /
			(currResConfidence * GetLuminance(fFromCanonOfNeigh) +
				static_cast<float>(neighCount) * neighRes.confidence * GetLuminance(neighSuffix.f));
		const float neighUcw = neighRes.wSum / GetLuminance(neighSuffix.f);
		const float neighRisWeight = neighMisWeight * pFromCanonOfNeigh * neighUcw; // pFromCanonOfNeigh includes pHat and jacobian

		// Stream neigh into res
		if (currRes.Update(SuffixPath(neighSuffix, currSuffix.lastPrefixIntSeed, fFromCanonOfNeigh), neighRisWeight, rng))
		{
			//printf("Spatial Suffix\n");
		}

		// Update canonical mis weight
		canonMisWeight += (currResConfidence * canonPHat) /
			((currResConfidence * canonPHat) +
				(neighRes.confidence * static_cast<float>(neighCount) * pFromNeighOfCanon));
	}

	canonMisWeight /= static_cast<float>(neighCount);

	// Calc canon ris weight
	const float canonRisWeight = canonMisWeight * currResWSum; // "pHat * ucw = wSum" here

	// Stream result of temporal reuse into reservoir again
	currRes.Update(currSuffix, canonRisWeight, rng);
}

extern "C" __global__ void __raygen__suffix_spatial_reuse()
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

	// Spatial suffix reuse
	SuffixSpatialReuse(pixelCoord);
}
