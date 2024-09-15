#include <cuda_runtime.h>
#include <optix_device.h>
#include <util/glm_cuda.h>
#include <graph/LaunchParams.h>
#include <graph/restir/path_gen.h>
#include <util/pixel_index.h>
#include <graph/restir/prefix_reuse.h>

__constant__ LaunchParams params;

static __forceinline__ __device__ bool PrefixSpatialReuse(const glm::uvec2& pixelCoord)
{
	// Assume: pixelCoord are valid

	// Get pixel index
	const uint32_t currPixelIdx = GetPixelIdx(pixelCoord, params);

	// Get current prefix
	Reservoir<PrefixPath>& currRes = params.restir.prefixReservoirs[2 * currPixelIdx + params.restir.frontBufferIdx];
	const PrefixPath currPrefix = currRes.sample;
	if (!currPrefix.IsValid()) { return false; }

	// Get canonical prefix
	const PrefixPath& canonPrefix = params.restir.canonicalPrefixes[currPixelIdx];
	if (!canonPrefix.IsValid() || !canonPrefix.primaryInt.IsValid()) { return false; }
	const float canonPHat = GetLuminance(canonPrefix.f);

	// Get rng
	PCG32& rng = params.restir.restirGBuffers[currPixelIdx].rng;

	// Cache data from current reservoir and reset
	const float currResWSum = currRes.wSum;
	const float currResConfidence = currRes.confidence;
	currRes.Reset();

	// RIS with pairwise MIS weights
	const int neighCount = params.restir.prefixSpatialCount;
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

		// Get neighbor res and prefix
		const Reservoir<PrefixPath>& neighRes = params.restir.prefixReservoirs[2 * neighPixelIdx + params.restir.backBufferIdx];
		const PrefixPath& neighPrefix = neighRes.sample;
		const bool skipBecauseOfNee = params.rendererType == RendererType::ConditionalRestir && neighPrefix.IsNee();
		if (!neighPrefix.IsValid() || skipBecauseOfNee) { continue; }

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
		const Reservoir<PrefixPath>& neighRes = params.restir.prefixReservoirs[2 * neighPixelIdx + params.restir.backBufferIdx];
		const PrefixPath& neighPrefix = neighRes.sample;

		// Get neighbor primary hit
		const Interaction neighPrimaryInt(neighPrefix.primaryInt, params.transforms);

		// Shift
		float jacobianNeighToCanon = 0.0f;
		const glm::vec3 fFromCanonOfNeigh = CalcCurrContribInOtherDomain(neighPrefix, currPrefix, jacobianNeighToCanon, params);
		const float pFromCanonOfNeigh = GetLuminance(fFromCanonOfNeigh) * jacobianNeighToCanon;

		float jacobianCanonToNeigh = 0.0f;
		const glm::vec3 fFromNeighOfCanon = CalcCurrContribInOtherDomain(currPrefix, neighPrefix, jacobianCanonToNeigh, params);
		const float pFromNeighOfCanon = GetLuminance(fFromNeighOfCanon) * jacobianCanonToNeigh;

		const glm::vec3& fFromCanonOfCanon = currPrefix.f;
		const glm::vec3& fFromNeighOfNeigh = neighPrefix.f;
		const float pFromCanonOfCanon = GetLuminance(fFromCanonOfCanon);
		const float pFromNeighOfNeigh = GetLuminance(fFromNeighOfNeigh);

		// Calc neigh mis weight
		float neighMisWeight = ComputeNeighborPairwiseMISWeight(
			fFromCanonOfNeigh, fFromNeighOfNeigh, jacobianNeighToCanon, k, currResConfidence, neighRes.confidence);
		if (glm::isnan(neighMisWeight) || glm::isinf(neighMisWeight)) neighMisWeight = 0.0f;

		// Calc neigh ris weight
		const float neighRisWeight = neighMisWeight * pFromCanonOfNeigh * neighRes.wSum;

		// Stream neigh into res
		if (currRes.Update(PrefixPath(neighPrefix, fFromCanonOfNeigh, currPrefix.primaryInt), neighRisWeight, rng, neighRes.confidence))
		{
			//printf("Spatial Prefix\n");
		}

		canonMisWeight += ComputeCanonicalPairwiseMISWeight(
			fFromCanonOfCanon, fFromNeighOfCanon, jacobianCanonToNeigh, k, currResConfidence, neighRes.confidence);
	}

	// Check canon mis weight
	if (glm::isinf(canonMisWeight) || glm::isnan(canonMisWeight)) { canonMisWeight = 0.0f; }

	// Calc canon ris weight
	const float canonRisWeight = canonMisWeight * currResWSum * GetLuminance(currPrefix.f);
	
	// Stream result of temporal reuse into reservoir again
	currRes.Update(currPrefix, canonRisWeight, rng, currResConfidence);
	
	// Finalize GRIS
	if (currRes.wSum > 0.0f)
	{
		currRes.wSum /= k + 1.0f;
		currRes.FinalizeGRIS(); 
	}

	return true;
}

extern "C" __global__ void __raygen__prefix_spatial_reuse()
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

	// Get rng
	const uint32_t pixelIdx = GetPixelIdx(pixelCoord, params);
	
	// Spatial prefix reuse
	glm::vec3 outputRadiance(0.0f);
	if (PrefixSpatialReuse(pixelCoord) && params.rendererType == RendererType::RestirPt)
	{
		// Get reservoir
		const Reservoir<PrefixPath>& prefixRes = params.restir.prefixReservoirs[2 * pixelIdx + params.restir.frontBufferIdx];
		const glm::vec3& f = prefixRes.sample.f;
		outputRadiance = prefixRes.wSum * f;
	}

	params.restir.prefixReservoirs[2 * pixelIdx + params.restir.backBufferIdx] =
		params.restir.prefixReservoirs[2 * pixelIdx + params.restir.frontBufferIdx];

	// Display if restir pt
	if (params.rendererType != RendererType::RestirPt || params.restir.spatialRoundIdx != params.restir.prefixSpatialRounds - 1) { return; }
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
