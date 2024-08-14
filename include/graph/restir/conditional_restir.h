#pragma once

#include <graph/path_functions.h>
#include <graph/luminance.h>
#include <graph/restir/subpath_reuse.h>

static __forceinline__ __device__ void ConditionalRestir(
	const glm::uvec3& launchIdx,
	const glm::vec3& origin, 
	const glm::vec3& dir, 
	PCG32& rng,
	LaunchParams& params)
{

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
