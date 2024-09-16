#include <cuda_runtime.h>
#include <optix_device.h>
#include <util/glm_cuda.h>
#include <util/random.h>
#include <graph/LaunchParams.h>
#include <util/pixel_index.h>
#include <graph/restir/path_gen.h>
#include <graph/restir/suffix_reuse.h>

__constant__ LaunchParams params;

static __forceinline__ __device__ void SuffixGenTempReuse(const glm::uvec2& pixelCoord)
{
	// Calc pixel index
	const uint32_t pixelIdx = GetPixelIdx(pixelCoord, params);

	// Get prefix
	const PrefixPath& prefix = params.restir.prefixReservoirs[pixelIdx * 2 + params.restir.frontBufferIdx].sample;
	if (!prefix.IsValid() || prefix.IsNee()) { return; }

	// Get rng
	PCG32& rng = params.restir.restirGBuffers[pixelIdx].rng;

	// Generate canonical suffix
	TraceSuffix(params.restir.canonicalSuffixes[pixelIdx], prefix, rng, params);
	const SuffixPath& canonSuffix = params.restir.canonicalSuffixes[pixelIdx];
	if (!canonSuffix.IsValid()) { return; }
	const float canonPHat = GetLuminance(canonSuffix.f);

	// Get current reservoir
	Reservoir<SuffixPath>& currRes = params.restir.suffixReservoirs[2 * pixelIdx + params.restir.frontBufferIdx];
	currRes.Reset();

	// Get prev pixel
	const glm::uvec2& prevPixelCoord = params.restir.restirGBuffers[pixelIdx].prevPixelCoord;
	const uint32_t prevPixelIdx = GetPixelIdx(prevPixelCoord, params);

	// Skip temporal reuse if prev pixel is invalid or temporal reuse is not active
	if (!params.restir.suffixEnableTemporal || !IsPixelValid(prevPixelCoord, params))
	{
		currRes.FinalizeGRIS();
		currRes.Update(canonSuffix, canonPHat / canonSuffix.p, rng, 1.0f);
		return;
	}

	// Get prev reservoir and prev suffix
	const Reservoir<SuffixPath>& prevRes = params.restir.suffixReservoirs[2 * prevPixelIdx + params.restir.backBufferIdx];
	const SuffixPath& prevSuffix = prevRes.sample;
	if (prevRes.wSum <= 0.0f || !prevSuffix.IsValid())
	{
		currRes.FinalizeGRIS();
		currRes.Update(canonSuffix, canonPHat / canonSuffix.p, rng, 1.0f);
		return;
	}

	// Temp reuse
	// Shift forward and backward
	float jacobianCanonToPrev = 0.0f;
	float jacobianPrevToCanon = 0.0f;
	const glm::vec3 fFromCanonOfPrev = CalcCurrContribInOtherDomain(prevSuffix, canonSuffix, jacobianPrevToCanon, params);
	const glm::vec3 fFromPrevOfCanon = CalcCurrContribInOtherDomain(canonSuffix, prevSuffix, jacobianCanonToPrev, params);

	// Calc talbot mis weights
	const float pFromCanonOfCanon = canonPHat;
	const float pFromCanonOfPrev = GetLuminance(fFromCanonOfPrev) * jacobianPrevToCanon;
	const float pFromPrevOfCanon = GetLuminance(fFromPrevOfCanon) * jacobianCanonToPrev;
	const float pFromPrevOfPrev = GetLuminance(prevSuffix.f);

	static constexpr float pairwiseK = 1.0f;
	const float prevMisWeight = ComputeNeighborPairwiseMISWeight(
		fFromCanonOfPrev, prevSuffix.f, jacobianPrevToCanon, pairwiseK, 1.0f, prevRes.confidence);
	const float canonMisWeight = 1.0f + ComputeCanonicalPairwiseMISWeight(
		canonSuffix.f, fFromPrevOfCanon, jacobianCanonToPrev, pairwiseK, 1.0f, prevRes.confidence);

	// Stream canonical sample
	const float canonRisWeight = canonMisWeight * canonPHat / canonSuffix.p;
	if (currRes.Update(canonSuffix, canonRisWeight, rng, 1.0f))
	{
		//printf("Curr Suffix\n");
	}

	// Stream prev samples
	const float prevUcw = prevRes.wSum * jacobianPrevToCanon / GetLuminance(prevSuffix.f);
	const float prevRisWeight = prevMisWeight * pFromCanonOfPrev * prevUcw;
	const SuffixPath shiftedPrevSuffix(prevSuffix, canonSuffix.lastPrefixInt, fFromCanonOfPrev);
	if (currRes.Update(shiftedPrevSuffix, prevRisWeight, rng, 1.0f))
	{
		//printf("Prev Suffix\n");
	}

	// Finalize GRIS
	if (currRes.wSum > 0.0f)
	{
		currRes.wSum /= pairwiseK + 1.0f;
		currRes.FinalizeGRIS(); 
	}
}

extern "C" __global__ void __raygen__suffix_gen_temp_reuse()
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

	//
	SuffixGenTempReuse(pixelCoord);

	// Swap buffers
	const uint32_t pixelIdx = GetPixelIdx(pixelCoord, params);
	params.restir.suffixReservoirs[2 * pixelIdx + params.restir.backBufferIdx] =
		params.restir.suffixReservoirs[2 * pixelIdx + params.restir.frontBufferIdx];
}
