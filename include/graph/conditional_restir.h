#pragma once

#include <graph/path_functions.h>

static constexpr __device__ Path CombinePrefixSuffix(const Path& prefix, const Path& suffix)
{
	Path path{};
	path.throughput = prefix.throughput * suffix.throughput;
	path.outputRadiance = prefix.throughput * suffix.outputRadiance;
	return path;
}

static constexpr __device__ Reservoir<Path>& GetPrefixReservoir(const uint32_t x, const uint32_t y, LaunchParams& params)
{
	return params.prefixReservoirs[y * params.width + x];
}

static constexpr __device__ Reservoir<Path>& GetSuffixReservoir(const uint32_t x, const uint32_t y, LaunchParams& params)
{
	return params.suffixReservoirs[y * params.width + x];
}

static constexpr __device__ Path ConditionalRestir(
	const glm::uvec3& launchIdx, 
	const glm::vec3& origin, 
	const glm::vec3& dir, 
	PCG32& rng,
	LaunchParams& params)
{
	// TODO: q' <- Temporal reprojection

	// Xp <- TraceNewPrefix(q)
	glm::vec3 suffixDir{};
	const Path prefix = SamplePrefix(origin, dir, rng, params, suffixDir);
	const glm::vec3 suffixOrigin = prefix.vertices[prefix.length];

	// TODO: Temporal prefix reuse using GRIS / Xp

	// Suffix reuse using CRIS
	// Xs <- TraceNewSuffix(Reservoirs[q].Xp)
	const Path suffix = SamplePath(suffixOrigin, suffixDir, MAX_PATH_LEN - prefix.length, rng, params);

	// Reservoirs[q].Xs <- CRIS(Xs, prevReservoirs[q'].Xs)
	Reservoir<Path>& suffixReservoir = { {}, 0.0f, 0, 0.0f };
	suffixReservoir.Update(suffix, 1.0f, rng);

	// Reservoirs[q].Xs <- SpatialSuffixReuse(Reservoirs)
	// prevReservoirs <- Reservoirs

	// Final gather
	// (Xp, Xs) <- TraceFullPath(q)
	// (R1, ..., Rk) <- FindSpatialKNN(Reservoirs, Xp, k)
	// Color[q] += ComputeMIS([Xp, R1.Xp, ..., Rk.Xp], Xs) * PathContrib(Xp, Xs) + Gather([R1, ..., Rk], Xp) / N

	// TODO: For loop

	// Simply test if combining prefix and suffix works
	return CombinePrefixSuffix(prefix, suffixReservoir.y);
}
