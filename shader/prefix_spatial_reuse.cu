#include <cuda_runtime.h>
#include <optix_device.h>
#include <util/glm_cuda.h>
#include <graph/LaunchParams.h>
#include <graph/restir/path_gen.h>
#include <util/pixel_index.h>
#include <graph/restir/prefix_reuse.h>

__constant__ LaunchParams params;

static constexpr uint32_t WINDOW_RADIUS = 3; // 48 total neighbors
static constexpr uint32_t WINDOW_SIZE = 2 * WINDOW_RADIUS + 1;

static constexpr __forceinline__ __device__ glm::uvec2 SelectSpatialNeighbor(const glm::uvec2& pixelCoord, PCG32& rng)
{
	const uint32_t xCoord = (pixelCoord.x - WINDOW_RADIUS) + (rng.NextUint32() % WINDOW_SIZE);
	const uint32_t yCoord = (pixelCoord.y - WINDOW_RADIUS) + (rng.NextUint32() % WINDOW_SIZE);
	return glm::uvec2(xCoord, yCoord);
}

static __forceinline__ __device__ void PrefixSpatialReuse(const glm::uvec2& pixelCoord, PCG32& rng)
{
	// Assume: pixelCoord are valid

	// Get pixel index
	const uint32_t currPixelIdx = GetPixelIdx(pixelCoord, params);

	// Get current prefix
	Reservoir<PrefixPath>& currRes = params.restir.prefixReservoirs[2 * currPixelIdx + params.restir.frontBufferIdx];
	const PrefixPath& currPrefix = currRes.sample;
	if (!currPrefix.IsValid()) { return; }

	// Get canonical prefix
	const PrefixPath& canonPrefix = params.restir.canonicalPrefixes[currPixelIdx];
	if (!canonPrefix.IsValid()) { return; }
	const float canonPHat = GetLuminance(canonPrefix.f);

	// Cache data from current reservoir and reset
	const float currResWSum = currRes.wSum;
	const float currResConfidence = currRes.confidence;
	currRes.Reset();

	// RIS with pairwise MIS weights
	const int neighCount = params.restir.prefixSpatialCount;

	uint32_t validNeighCount = 0;
	float canonMisWeight = 0.0f;

	for (uint32_t neighIdx = 0; neighIdx < neighCount; ++neighIdx)
	{
		// Select new neighbor
		const glm::uvec2 neighPixelCoord = SelectSpatialNeighbor(pixelCoord, rng);
		if (!IsPixelValid(neighPixelCoord, params)) { continue; }
		const uint32_t neighPixelIdx = GetPixelIdx(neighPixelCoord, params);

		// Get neighbor res and prefix
		const Reservoir<PrefixPath>& neighRes = params.restir.prefixReservoirs[2 * neighPixelIdx + params.restir.frontBufferIdx];
		const PrefixPath& neighPrefix = neighRes.sample;
		if (!neighPrefix.IsValid() || neighPrefix.IsNee()) { continue; }

		// Get neighbor primary hit
		Interaction neighPrimaryInt{};
		TraceInteractionSeed(neighPrefix.primaryIntSeed, neighPrimaryInt, params);
		if (!neighPrimaryInt.valid) { continue; }
		++validNeighCount;

		// Shift
		float jacobianNeighToCanon = 0.0f;
		const glm::vec3 fFromCanonOfNeigh = CalcCurrContribInOtherDomain(neighPrefix, canonPrefix, jacobianNeighToCanon, params);
		const float pFromCanonOfNeigh = GetLuminance(fFromCanonOfNeigh) * jacobianNeighToCanon;

		float jacobianCanonToNeigh = 0.0f;
		const glm::vec3 fFromNeighOfCanon = CalcCurrContribInOtherDomain(canonPrefix, neighPrefix, jacobianCanonToNeigh, params);
		const float pFromNeighOfCanon = GetLuminance(fFromNeighOfCanon) * jacobianCanonToNeigh;

		// Calc neigh mis weight
		const float neighMisWeight = neighRes.confidence * GetLuminance(neighPrefix.f) /
			(currResConfidence * GetLuminance(fFromCanonOfNeigh) + 
				static_cast<float>(neighCount) * neighRes.confidence * GetLuminance(neighPrefix.f));
		const float neighUcw = neighRes.wSum / GetLuminance(neighPrefix.f);
		const float neighRisWeight = neighMisWeight * pFromCanonOfNeigh * neighUcw; // pFromCanonOfNeigh includes pHat and jacobian

		// Stream neigh into res
		if (currRes.Update(PrefixPath(neighPrefix, fFromCanonOfNeigh, currPrefix.primaryIntSeed), neighRisWeight, rng))
		{
			//printf("Spatial Prefix\n");
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
	currRes.Update(currPrefix, canonRisWeight, rng);
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
	const uint32_t pixelIdx = launchIdx.y * launchDims.x + launchIdx.x;
	PCG32& rng = params.restir.restirGBuffers[pixelIdx].rng;

	// Spatial prefix reuse
	PrefixSpatialReuse(pixelCoord, rng);
}
