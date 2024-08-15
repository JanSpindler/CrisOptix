#pragma once

#include <glm/glm.hpp>
#include <util/random.h>
#include <graph/luminance.h>
#include <graph/restir/struct/LastVertexState.h>
#include <graph/restir/struct/RestirPathFlags.h>

struct PathReservoir
{
	float M;
	float wSum;
	glm::vec3 integrand;
    RestirPathFlags pathFlags;
	PCG32 initRng;
	PCG32 reconRng;
	PCG32 suffixInitRng;
	float lightPdf;
	// HitInfo
	float reconJacobian;
	glm::vec3 reconThroughput;

    __forceinline__ __device__ __host__ PathReservoir() :
        M(0.0f),
        wSum(0.0f),
        integrand(0.0f),
        initRng(),
        reconRng(),
        suffixInitRng(),
        pathFlags(0),
        lightPdf(0.0f),
        reconJacobian(0.0f),
        reconThroughput(0.0f)
    {
    }

	__forceinline__ __device__ __host__ void Init()
	{
		M = 0.0f;
		wSum = 0.0f;
		integrand = glm::vec3(0.0f);
		initRng = PCG32();
		reconRng = PCG32();
		suffixInitRng = PCG32();
        pathFlags.flags = 0;
	}

	__forceinline__ __device__ __host__ bool MergeSameDomain(const PathReservoir& inRes, PCG32& rng, const float misWeight)
	{
		float weight = GetLuminance(inRes.integrand) * inRes.wSum * misWeight;
		if (glm::isnan(weight) || glm::isinf(weight)) { weight = 0.0f; }

		M += inRes.M;
		wSum += weight;

		if (rng.NextFloat() * wSum < weight)
		{
			integrand = inRes.integrand;
            pathFlags.flags = inRes.pathFlags.flags;

			initRng = inRes.initRng;
			reconRng = inRes.reconRng;
			suffixInitRng = inRes.suffixInitRng;

			lightPdf = inRes.lightPdf;
			reconJacobian = inRes.reconJacobian;
			reconThroughput = inRes.reconThroughput;

			return true;
		}

		return false;
	}

	__forceinline__ __device__ __host__ bool Merge(const glm::vec3& inIntegrand, const float inJacobian, const PathReservoir& inRes, PCG32& rng, const float misWeight)
	{
		float weight = GetLuminance(inIntegrand) * inJacobian * inRes.wSum * misWeight;
		if (glm::isnan(weight) || glm::isinf(weight)) { weight = 0.0f; }

		M += inRes.M;
		wSum += weight;

		if (rng.NextFloat() * wSum < weight)
		{
			integrand = inIntegrand;
            pathFlags.flags = inRes.pathFlags.flags;

			initRng = inRes.initRng;
			reconRng = inRes.reconRng;
			suffixInitRng = inRes.suffixInitRng;

			lightPdf = inRes.lightPdf;
			reconJacobian = inRes.reconJacobian;
			reconThroughput = inRes.reconThroughput;

			return true;
		}

		return false;
	}
};
