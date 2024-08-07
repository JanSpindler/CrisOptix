#pragma once

#include <graph/LaunchParams.h>
#include <graph/Reservoir.h>
#include <graph/sample_emitter.h>

static constexpr __device__ float GetPHatDi(const SurfaceInteraction& interaction, const EmitterSample& emitterSample, PCG32& rng)
{
	const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const SurfaceInteraction&, const glm::vec3&>(
		interaction.meshSbtData->evalMaterialSbtIdx,
		interaction,
		glm::normalize(emitterSample.pos - interaction.pos));
	const float pHat = glm::length(brdfEvalResult.brdfResult);
	return pHat;
}

static constexpr __device__ Reservoir<EmitterSample>& GetDiReservoir(const uint32_t x, const uint32_t y, LaunchParams& params)
{
	return params.diReservoirs[y * params.width + x];
}

static constexpr __device__ Reservoir<EmitterSample> CombineReservoirDi(
	const Reservoir<EmitterSample>& r1,
	const Reservoir<EmitterSample>& r2,
	const SurfaceInteraction& interaction,
	PCG32& rng)
{
	const float pHat1 = GetPHatDi(interaction, r1.y, rng);
	const float pHat2 = GetPHatDi(interaction, r2.y, rng);

	Reservoir<EmitterSample> res = { {}, 0.0f, 0 };
	res.Update(r1.y, pHat1 * r1.W * r1.M, rng);
	res.Update(r2.y, pHat2 * r2.W * r2.M, rng);

	res.M = r1.M + r2.M;
	res.W = GetPHatDi(interaction, res.y, rng) * res.wSum / static_cast<float>(res.M);
}

static constexpr __device__ Reservoir<EmitterSample> RestirRis(const SurfaceInteraction& interaction, const size_t sampleCount, PCG32& rng, const LaunchParams& params)
{
	Reservoir<EmitterSample> reservoir = { {}, 0.0f, 0 };

	for (size_t idx = 0; idx < sampleCount; ++idx)
	{
		const EmitterSample emitterSample = SampleEmitter(rng, params);
		const float pHat = GetPHatDi(interaction, emitterSample, rng);
		reservoir.Update(emitterSample, pHat / emitterSample.p, rng);
	}

	return reservoir;
}

static constexpr __device__ void RestirDi(
	const glm::uvec3& launchIdx,
	const SurfaceInteraction& interaction,
	PCG32& rng,
	LaunchParams& params)
{
	// Generate new samples
	Reservoir<EmitterSample> newReservoir = RestirRis(interaction, params.restirParams.canonicalCount, rng, params);

	// Check if shadowed
	const bool occluded = TraceOcclusion(
		params.traversableHandle,
		interaction.pos,
		glm::normalize(newReservoir.y.pos - interaction.pos),
		1e-3f,
		glm::length(newReservoir.y.pos - interaction.pos),
		params.occlusionTraceParams);
	if (occluded) { newReservoir.W = 0.0f; }

	// Temporal reuse
	if (params.frameIdx > 1 && params.restirParams.enableTemporal)
	{
		const glm::vec2 oldUV = params.cameraData.prevW2V * glm::vec4(interaction.pos, 1.0f);
		if (oldUV.x == glm::clamp(oldUV.x, 0.0f, 1.0f) && oldUV.y == glm::clamp(oldUV.y, 0.0f, 1.0f))
		{
			newReservoir = CombineReservoirDi(newReservoir, GetDiReservoir(launchIdx.x, launchIdx.y, params), interaction, rng);
		}
	}

	// Spatial reuse
	if (params.restirParams.enableSpatial)
	{
		const size_t N = params.restirParams.spatialKernelSize;
		for (size_t n = 0; n < params.restirParams.spatialCount; ++n)
		{
			const size_t nX = launchIdx.x + (rng.NextUint32() % (2 * N + 1)) - N;
			const size_t nY = launchIdx.y + (rng.NextUint32() % (2 * N + 1)) - N;
			if (nX < params.width && nY < params.height)
			{
				newReservoir = CombineReservoirDi(newReservoir, GetDiReservoir(nX, nY, params), interaction, rng);
			}
		}
	}

	// Store reservoir
	GetDiReservoir(launchIdx.x, launchIdx.y, params) = newReservoir;
}
