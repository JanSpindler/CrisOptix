#include <cuda_runtime.h>
#include <optix_device.h>
#include <util/glm_cuda.h>
#include <graph/LaunchParams.h>
#include <graph/restir/path_gen.h>
#include <util/pixel_index.h>
#include <graph/restir/prefix_reuse.h>

__constant__ LaunchParams params;

static __forceinline__ __device__ bool PrefixGenTempReuse(
	const glm::uvec2& pixelCoord,
	const glm::uvec2& prevPixelCoord,
	const glm::vec3& origin, 
	const glm::vec3& dir, 
	PCG32& rng)
{
	// Get pixel index
	const size_t pixelIdx = GetPixelIdx(pixelCoord, params);

	// Generate canonical prefix and store in buffer for later usage with spatial reuse
	TracePrefix(params.restir.canonicalPrefixes[pixelIdx], origin, dir, rng, params);
	const PrefixPath& canonPrefix = params.restir.canonicalPrefixes[pixelIdx];
	if (!canonPrefix.IsValid()) { return false; }
	const float canonPHat = GetLuminance(canonPrefix.f);

	// Get current reservoir and reset it
	Reservoir<PrefixPath>& currRes = params.restir.prefixReservoirs[2 * pixelIdx + params.restir.frontBufferIdx];
	currRes.Reset();

	// If no temporal reuse or prev pixel is invalid
	if (!params.restir.prefixEnableTemporal || !IsPixelValid(prevPixelCoord, params))
	{
		// Skip temporal reuse
		currRes.Update(canonPrefix, canonPHat / canonPrefix.p, rng);
		currRes.FinalizeGRIS();
		return true;
	}

	// Get prev reservoir and prev prefix
	const uint32_t prevPixelIdx = GetPixelIdx(prevPixelCoord, params);
	const Reservoir<PrefixPath>& prevRes = params.restir.prefixReservoirs[2 * pixelIdx + params.restir.backBufferIdx];
	const PrefixPath& prevPrefix = prevRes.sample;

	// If ...
	const bool skipBecauseOfNee = params.rendererType == RendererType::ConditionalRestir && prevPrefix.IsNee();
	if (prevRes.wSum <= 0.0f || // Do not reuse prefixes with 0 ucw
		!prevPrefix.IsValid() || // Do not reuse invalid prefixes
		skipBecauseOfNee) // Do not reuse prefixes that wont generate suffixes
	{
		currRes.Update(canonPrefix, canonPHat / canonPrefix.p, rng);
		currRes.FinalizeGRIS();
		return true;
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

	const float canonMisWeight = 1.0f * pFromCanonOfCanon / (1.0f * pFromCanonOfCanon + prevRes.confidence * pFromPrevOfCanon);
	const float prevMisWeight = prevRes.confidence * pFromPrevOfPrev / (1.0f * pFromCanonOfPrev + prevRes.confidence * pFromPrevOfPrev);
	
	// Stream canonical sample
	const float canonRisWeight = canonMisWeight * canonPHat / canonPrefix.p;
	if (currRes.Update(canonPrefix, canonRisWeight, rng))
	{
		//printf("Curr Prefix\n");
	}

	// Stream prev samples
	const float prevRisWeight = prevMisWeight * pFromCanonOfPrev * prevRes.wSum;
	const PrefixPath shiftedPrevPrefix(prevPrefix, fFromCanonOfPrev, canonPrefix.primaryInt);
	if (currRes.Update(shiftedPrevPrefix, prevRisWeight, rng))
	{
		//printf("Prev Prefix\n");
	}

	// Finalize GRIS
	if (currRes.wSum > 0.0f) { currRes.FinalizeGRIS(); }

	return true;
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
	if (params.rendererType == RendererType::ConditionalRestir || params.rendererType == RendererType::RestirPt)
	{
		// Get motion vector
		const size_t pixelIdx = GetPixelIdx(pixelCoord, params);
		const glm::vec2 motionVector = params.motionVectors[pixelIdx];

		// Calc prev pixel coord
		const glm::uvec2 prevPixelCoord = glm::uvec2(glm::vec2(pixelCoord) + glm::vec2(0.5f) + motionVector);

		//
		const bool validPrimaryInt = PrefixGenTempReuse(pixelCoord, prevPixelCoord, origin, dir, rng);
		if (params.rendererType == RendererType::RestirPt && !params.restir.prefixEnableSpatial)
		{
			// Get reservoir
			if (validPrimaryInt)
			{
				const Reservoir<PrefixPath>& prefixRes = params.restir.prefixReservoirs[2 * pixelIdx + params.restir.frontBufferIdx];
				const glm::vec3& f = prefixRes.sample.f;
				outputRadiance = prefixRes.wSum * f;
			}

			// Display if restir pt
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

		// Copy front to back reservoir for spatial reuse
		if ((params.rendererType == RendererType::RestirPt && params.restir.prefixEnableSpatial) ||
			(params.rendererType == RendererType::ConditionalRestir && params.restir.suffixEnableSpatial))
		{
			params.restir.prefixReservoirs[2 * pixelIdx + params.restir.backBufferIdx] = 
				params.restir.prefixReservoirs[2 * pixelIdx + params.restir.frontBufferIdx];
		}

		// Store restir g buffer
		params.restir.restirGBuffers[pixelIdx] = RestirGBuffer(prevPixelCoord, rng, validPrimaryInt);
	}
	// If not restir
	else if (params.rendererType == RendererType::PathTracer)
	{
		// Perform normal path tracing
		outputRadiance = TraceCompletePath(origin, dir, rng, params);
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
