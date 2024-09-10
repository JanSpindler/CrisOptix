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
	float canonMisWeight = 0.0f;
	uint32_t validNeighCount = 0;

	for (uint32_t neighIdx = 0; neighIdx < neighCount; ++neighIdx)
	{
		// Select new neighbor
		const glm::uvec2 neighPixelCoord = SelectSpatialNeighbor(pixelCoord, rng);
		if (!IsPixelValid(neighPixelCoord, params)) { continue; }
		const uint32_t neighPixelIdx = GetPixelIdx(neighPixelCoord, params);

		//
		if (!params.restir.restirGBuffers[neighPixelIdx].primaryIntValid) { continue; }

		// Get neighbor res and prefix
		const Reservoir<PrefixPath>& neighRes = params.restir.prefixReservoirs[2 * neighPixelIdx + params.restir.frontBufferIdx];
		const PrefixPath& neighPrefix = neighRes.sample;
		const bool skipBecauseOfNee = params.rendererType == RendererType::ConditionalRestir && neighPrefix.IsNee();
		if (!neighPrefix.IsValid() || skipBecauseOfNee) { continue; }

		// Get neighbor primary hit
		const Interaction neighPrimaryInt(neighPrefix.primaryInt, params.transforms);
		if (!neighPrimaryInt.valid) { continue; }

		// Shift
		float jacobianNeighToCanon = 0.0f;
		const glm::vec3 fFromCanonOfNeigh = CalcCurrContribInOtherDomain(neighPrefix, canonPrefix, jacobianNeighToCanon, params);
		const float pFromCanonOfNeigh = GetLuminance(fFromCanonOfNeigh) * jacobianNeighToCanon;

		float jacobianCanonToNeigh = 0.0f;
		const glm::vec3 fFromNeighOfCanon = CalcCurrContribInOtherDomain(canonPrefix, neighPrefix, jacobianCanonToNeigh, params);
		const float pFromNeighOfCanon = GetLuminance(fFromNeighOfCanon) * jacobianCanonToNeigh;

		//
		++validNeighCount;

		// Calc neigh mis weight
		const float neighMisWeight = neighRes.confidence * GetLuminance(neighPrefix.f) /
			(currResConfidence * GetLuminance(fFromCanonOfNeigh) + 
				static_cast<float>(neighCount) * neighRes.confidence * GetLuminance(neighPrefix.f));
		const float neighUcw = neighRes.wSum / GetLuminance(neighPrefix.f);
		const float neighRisWeight = neighMisWeight * pFromCanonOfNeigh * neighUcw; // pFromCanonOfNeigh includes pHat and jacobian

		// Stream neigh into res
		if (currRes.Update(PrefixPath(neighPrefix, fFromCanonOfNeigh, currPrefix.primaryInt), neighRisWeight, rng))
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
		outputRadiance = prefixRes.wSum * f / GetLuminance(f);
	}

	// Display if restir pt
	if (params.rendererType != RendererType::RestirPt) { return; }
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
