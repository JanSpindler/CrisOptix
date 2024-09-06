#include <cuda_runtime.h>
#include <optix_device.h>
#include <util/glm_cuda.h>
#include <graph/LaunchParams.h>
#include <graph/restir/path_gen.h>
#include <util/pixel_index.h>
#include <graph/restir/ris_helper.h>
#include <graph/restir/prefix_reuse.h>

__constant__ LaunchParams params;

static __forceinline__ __device__ void PrefixGenTempReuse(
	const glm::uvec2& pixelCoord,
	const glm::uvec2& prevPixelCoord,
	const glm::vec3& origin, 
	const glm::vec3& dir, 
	PCG32& rng)
{
	// Get pixel index
	const size_t pixelIdx = GetPixelIdx(pixelCoord, params);
	const size_t prevPixelIdx = GetPixelIdx(prevPixelCoord, params);

	// Generate canonical prefix and store in buffer for later usage with spatial reuse
	params.restir.canonicalPrefixes[pixelIdx] = TracePrefix(origin, dir, params.restir.minPrefixLen, rng, params);
	const PrefixPath& canonPrefix = params.restir.canonicalPrefixes[pixelIdx];
	const float canonPHat = GetLuminance(canonPrefix.f);

	// Get current reservoir and reset it
	Reservoir<PrefixPath>& currRes = params.restir.prefixReservoirs[2 * pixelIdx + params.restir.frontBufferIdx];
	currRes.Reset();

	// Get prev reservoir and prev prefix
	Reservoir<PrefixPath>& prevRes = params.restir.prefixReservoirs[2 * pixelIdx + params.restir.backBufferIdx];
	const PrefixPath& prevPrefix = prevRes.sample;

	// If ...
	if (!params.restir.prefixEnableTemporal || // Do not reuse when not wanted
		!IsPixelValid(prevPixelCoord, params) || // Do not reuse from invalid pixels
		prevRes.wSum <= 0.0f || // Do not reuse prefixes with 0 ucw
		!prevPrefix.IsValid() || // Do not reuse invalid prefixes
		prevPrefix.IsNee()) // Do not reuse prefixes that wont generate suffixes
	{
		// Store canonical prefix and end
		const float risWeight = canonPHat / canonPrefix.p;
		currRes.Update(canonPrefix, risWeight, rng);
		return;
	}

	// Temp reuse
	// Shift forward and backward
	float jacobianCanonToPrev = 0.0f;
	float jacobianPrevToCanon = 0.0f;
	const glm::vec3 fFromCanonOfPrev = CalcCurrContribInOtherDomain(prevPrefix, canonPrefix, jacobianPrevToCanon, params);
	const glm::vec3 fFromPrevOfCanon = CalcCurrContribInOtherDomain(canonPrefix, prevPrefix, jacobianCanonToPrev, params);

	// Calc talbot mis weights
	const float pFromCanonOfCanon = canonPHat;
	const float pFromCanonOfPrev = GetLuminance(fFromCanonOfPrev) * jacobianPrevToCanon;
	const float pFromPrevOfCanon = GetLuminance(fFromPrevOfCanon) * jacobianCanonToPrev;
	const float pFromPrevOfPrev = GetLuminance(prevPrefix.f);

	const float canonMisWeight = pFromCanonOfCanon / (pFromCanonOfCanon + pFromPrevOfCanon);
	const float prevMisWeight = pFromPrevOfPrev / (pFromCanonOfPrev + pFromPrevOfPrev);

	// Stream canonical sample
	const float canonRisWeight = canonMisWeight * canonPHat / canonPrefix.p;
	currRes.Update(canonPrefix, canonRisWeight, rng);

	// Stream prev samples
	const float prevUcw = prevRes.wSum * jacobianPrevToCanon / GetLuminance(prevPrefix.f);
	const float prevRisWeight = prevMisWeight * pFromCanonOfPrev * prevUcw;
	const PrefixPath shiftedPrevPrefix(prevPrefix, glm::vec3(0.0f), canonPrefix.primaryIntSeed);
	currRes.Update(shiftedPrevPrefix, prevRisWeight, rng);
}

extern "C" __global__ void __raygen__prefix_gen_temp_reuse()
{
	//
	const glm::uvec3 launchIdx = cuda2glm(optixGetLaunchIndex());
	const glm::uvec3 launchDims = cuda2glm(optixGetLaunchDimensions());
	const glm::uvec2 pixelCoord = glm::uvec2(launchIdx);
	const size_t pixelIdx = GetPixelIdx(pixelCoord, params);

	// Exit if invalid launch idx
	if (launchIdx.x >= params.width || launchIdx.y >= params.height || launchIdx.z >= 1)
	{
		return;
	}

	// Init RNG
	const uint64_t seed = SampleTEA64(pixelIdx, params.random);
	PCG32 rng(seed);

	// Init radiance with 0
	glm::vec3 outputRadiance(0.0f);

	// Spawn camera ray
	glm::vec3 origin(0.0f);
	glm::vec3 dir(0.0f);
	glm::vec2 uv = (glm::vec2(launchIdx) + rng.Next2d()) / glm::vec2(params.width, params.height);
	uv = 2.0f * uv - 1.0f; // [0, 1] -> [-1, 1]
	SpawnCameraRay(params.cameraData, uv, origin, dir);

	// If restir
	if (params.enableRestir)
	{
		// Get motion vector
		const size_t pixelIdx = GetPixelIdx(pixelCoord, params);
		const glm::vec2 motionVector = params.motionVectors[pixelIdx];

		// Calc prev pixel coord
		const glm::uvec2 prevPixelCoord = glm::uvec2(glm::vec2(pixelCoord) + glm::vec2(0.5f) + motionVector);

		//
		PrefixGenTempReuse(pixelCoord, prevPixelCoord, origin, dir, rng);

		// Store restir g buffer
		params.restir.restirGBuffers[pixelIdx] = RestirGBuffer(prevPixelCoord, rng);
	}
	// If not restir
	else
	{
		// Perform normal path tracing
		outputRadiance = TraceCompletePath(origin, dir, 8, rng, params);
		if (glm::any(glm::isnan(outputRadiance) || glm::isinf(outputRadiance))) { outputRadiance = glm::vec3(0.0f); }

		// Store radiance output
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
}
