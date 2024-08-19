#pragma once

#include <graph/luminance.h>
//#include <graph/restir/pass/prefix_produce_retrace.h>
//#include <graph/restir/pass/prefix_retrace.h>
//#include <graph/restir/pass/prefix_resampling.h>
#include <graph/restir/pass/trace_new_suffixes.h>
#include <graph/restir/pass/temporal_suffix_reuse.h>
#include <graph/restir/pass/spatial_suffix_reuse.h>

static __forceinline__ __device__ glm::vec3 ConditionalRestir(
	const glm::uvec2& pixelCoord,
	const glm::vec3& origin,
	const glm::vec3& dir,
	PCG32& rng,
	LaunchParams& params)
{
	//SuffixPath testPath{};
	//GenSuffix(testPath, origin, dir, 8, 0.5f, 8, rng, params);
	//return testPath.radiance;

	// Sample surface interaction
	SurfaceInteraction primaryInteraction{};
	TraceWithDataPointer<SurfaceInteraction>(
		params.traversableHandle,
		origin,
		dir,
		1e-3f,
		1e16f,
		params.surfaceTraceParams,
		&primaryInteraction);
	if (!primaryInteraction.valid) { return glm::vec3(0.0f); }

	// Pixel idx
	const size_t pixelIdx = GetPixelIdx(pixelCoord, params);

	// Prev pixel coord
	const glm::vec2 motionVector = glm::vec2(0.0f);// params.motionVectors[pixelIdx];
	const glm::uvec2 prevPixelCoord = pixelCoord;/*glm::uvec2(
		glm::vec2(pixelCoord) +
		motionVector * glm::vec2(params.width, params.height) +
		glm::vec2(0.5f));*/
	const size_t prevPixelIdx = GetPixelIdx(prevPixelCoord, params);
	
	// Gen canonical prefix
	Reservoir<PrefixPath> prefixRes{};
	PrefixPath canonPrefix{};
	GenPrefix(canonPrefix, origin, dir, params.restir.minPrefixLen, 8, rng, params);

	// Stream canonical prefix into prefix reservoir
	prefixRes.Update(canonPrefix, GetLuminance(canonPrefix.throughput) / canonPrefix.p, canonPrefix.throughput, rng);

	// Prefix resampling using cris
	//if (params.restir.adaptivePrefixLength)
	//{
	//	PrefixProduceRetrace(pixelIdx, prevPixelCoord, centralRes, primaryInteraction, params);
	//	PrefixRetrace(pixelIdx, prevPixelCoord, centralRes, primaryInteraction, rng, params);
	//}
	//PrefixResampling();

	// End if prefix has performed nee
	if (prefixRes.currentSample.nee) { return glm::vec3(prefixRes.currentIntegrand); }

	// Check if prefix is valid
	if (!prefixRes.currentSample.valid || !prefixRes.currentSample.lastInteraction.valid) { return glm::vec3(0.0f); }

	// Trace New Suffixes
	Reservoir<SuffixPath> suffixRes{};
	Reconnection recon{};
	TraceNewSuffixes(pixelIdx, prefixRes.currentSample, suffixRes, recon, rng, params);

	// Temporal suffix reuse
	if (params.restir.suffixEnableTemporal)
	{
		TemporalSuffixReuse(pixelIdx, prevPixelCoord, prefixRes.currentSample, suffixRes, recon, rng, params);
	}

	// Spatial suffix reuse
	if (params.restir.suffixEnableSpatial)
	{
		//SpatialSuffixReuse(pixelCoord, rng, params);
	}

	// Store prefix and suffix reservoirs
	params.restir.prefixReservoirs[pixelIdx] = prefixRes;
	params.restir.suffixReservoirs[pixelIdx] = suffixRes;

	// Shade
	return prefixRes.currentIntegrand * suffixRes.currentIntegrand * recon.GetWeight3f();

	// TODO: Final gather
	//for (integrationPrefixCount)
	//{
	//	// Trace New Prefixes
	//	TraceNewPrefixes();

	//	// Final Gather
	//	// Final Gather Neighbor Search
	//	PrefixNeighborSearch();
	//	if (useTalbotMisForGather)
	//	{
	//		SuffixProduceRetraceTalbot();
	//		SuffixRetraceTalbot();
	//	}
	//	else
	//	{
	//		SuffixProduceRetrace();
	//		SuffixRetrace();
	//	}
	//
	//	// Final Gather Integration
	//	SuffixResampling();
	//}
}

//static constexpr __device__ void ConditionalRestir(
//	const glm::uvec3& launchIdx, 
//	const glm::vec3& origin, 
//	const glm::vec3& dir, 
//	PCG32& rng,
//	LaunchParams& params)
//{
//	// TODO: q' <- Temporal reprojection
//
//	// Xp <- TraceNewPrefix(q)
//
//	// TODO: Temporal prefix reuse using GRIS / Xp
//
//	// Suffix reuse using CRIS
//	// Xs <- TraceNewSuffix(Reservoirs[q].Xp)
//
//	// Reservoirs[q].Xs <- CRIS(Xs, prevReservoirs[q'].Xs)
//
//	// Reservoirs[q].Xs <- SpatialSuffixReuse(Reservoirs)
//	
//	// prevReservoirs <- Reservoirs
//
//	// Final gather
//	// (Xp, Xs) <- TraceFullPath(q)
//	// (R1, ..., Rk) <- FindSpatialKNN(Reservoirs, Xp, k)
//	// Color[q] += ComputeMIS([Xp, R1.Xp, ..., Rk.Xp], Xs) * PathContrib(Xp, Xs) + Gather([R1, ..., Rk], Xp) / N
//
//	// TODO: For loop
//
//	// Simply test if combining prefix and suffix works
//}
