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
	const SuffixPath currSuffix = currRes.sample;
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
	const int neighCount = params.restir.suffixSpatialCount;
	float canonMisWeight = 1.0f;
	uint32_t validNeighCount = 0;
	uint32_t validNeighFlags = 0;

	// Count valid neighbors
	glm::uvec2 neighPixelCoords[MAX_SPATIAL_NEIGH_COUNT];
	for (uint32_t neighIdx = 0; neighIdx < neighCount; ++neighIdx)
	{
		// Select new neighbor
		neighPixelCoords[neighIdx] = SelectSpatialNeighbor(pixelCoord, rng);
		const glm::uvec2& neighPixelCoord = neighPixelCoords[neighIdx];
		if (!IsPixelValid(neighPixelCoord, params)) { continue; }
		const uint32_t neighPixelIdx = GetPixelIdx(neighPixelCoord, params);

		// Check neighbor primary interaction
		if (!params.restir.restirGBuffers[neighPixelIdx].primaryIntValid) { continue; }

		// Get neighbor res and suffix
		const Reservoir<SuffixPath>& neighRes = params.restir.suffixReservoirs[2 * neighPixelIdx + params.restir.backBufferIdx];
		const SuffixPath& neighSuffix = neighRes.sample;
		if (!neighSuffix.IsValid()) { continue; }

		// Mark neigh as valid
		++validNeighCount;
		validNeighFlags |= 1 << neighIdx;
	}
	
	// Perform reuse with valid neighbors
	const float validNeighCountF = static_cast<float>(validNeighCount);
	const float k = validNeighCountF;
	for (uint32_t neighIdx = 0; neighIdx < neighCount; ++neighIdx)
	{
		// Skip if not marked as valid
		if (!((validNeighFlags >> neighIdx) & 1)) { continue; }

		// Select new neighbor
		const glm::uvec2& neighPixelCoord = neighPixelCoords[neighIdx];
		const uint32_t neighPixelIdx = GetPixelIdx(neighPixelCoord, params);

		// Get neighbor res and prefix
		const Reservoir<SuffixPath>& neighRes = params.restir.suffixReservoirs[2 * neighPixelIdx + params.restir.frontBufferIdx];
		const SuffixPath& neighSuffix = neighRes.sample;
		
		// Get neighbor primary hit
		const Interaction neighLastPrefixInt(neighSuffix.lastPrefixInt, params.transforms);
		
		// Shift
		float jacobianNeighToCanon = 0.0f;
		const glm::vec3 fFromCanonOfNeigh = CalcCurrContribInOtherDomain(neighSuffix, currSuffix, jacobianNeighToCanon, params);
		const float pFromCanonOfNeigh = GetLuminance(fFromCanonOfNeigh) * jacobianNeighToCanon;

		float jacobianCanonToNeigh = 0.0f;
		const glm::vec3 fFromNeighOfCanon = CalcCurrContribInOtherDomain(currSuffix, neighSuffix, jacobianCanonToNeigh, params);
		const float pFromNeighOfCanon = GetLuminance(fFromNeighOfCanon) * jacobianCanonToNeigh;

		const glm::vec3& fFromCanonOfCanon = currSuffix.f;
		const glm::vec3& fFromNeighOfNeigh = neighSuffix.f;
		const float pFromCanonOfCanon = GetLuminance(fFromCanonOfCanon);
		const float pFromNeighOfNeigh = GetLuminance(fFromNeighOfNeigh);

		// Calc neigh mis weight
		float neighMisWeight = ComputeNeighborPairwiseMISWeight(
			fFromCanonOfNeigh, fFromNeighOfNeigh, jacobianNeighToCanon, k, currResConfidence, neighRes.confidence);
		if (glm::isnan(neighMisWeight) || glm::isinf(neighMisWeight)) neighMisWeight = 0.0f;

		// Calc neigh ris weight
		const float neighRisWeight = neighMisWeight * pFromCanonOfNeigh * neighRes.wSum;

		// Stream neigh into res
		if (currRes.Update(SuffixPath(neighSuffix, currSuffix.lastPrefixInt, fFromCanonOfNeigh), neighRisWeight, rng, neighRes.confidence))
		{
			//printf("Spatial Prefix\n");
		}

		canonMisWeight += ComputeCanonicalPairwiseMISWeight(
			fFromCanonOfCanon, fFromNeighOfCanon, jacobianCanonToNeigh, k, currResConfidence, neighRes.confidence);
	}

	// Check canon mis weight
	if (glm::isinf(canonMisWeight) || glm::isnan(canonMisWeight)) { canonMisWeight = 0.0f; }

	// Calc canon ris weight
	const float canonRisWeight = canonMisWeight * currResWSum * GetLuminance(currSuffix.f);

	// Stream result of temporal reuse into reservoir again
	currRes.Update(currSuffix, canonRisWeight, rng, currResConfidence);

	// Finalize GRIS
	if (currRes.wSum > 0.0f) 
	{
		currRes.wSum /= k + 1.0f;
		currRes.FinalizeGRIS(); 
	}

	// Store in back buffer
	params.restir.suffixReservoirs[2 * currPixelIdx + params.restir.backBufferIdx] =
		params.restir.suffixReservoirs[2 * currPixelIdx + params.restir.frontBufferIdx];
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
