#pragma once

#include <glm/glm.hpp>
#include <util/random.h>
#include <graph/luminance.h>
#include <graph/restir/LastVertexState.h>

struct RestirPathFlags
{
    uint32_t flags = 0;

    __forceinline__ __device__ __host__ RestirPathFlags(const uint32_t _flags = 0) :
        flags(_flags)
    {
    }

    // assuming data contains from bit 0 to bit 3, isDelta, isTransmission, and BSDFComponent
    __forceinline__ __device__ __host__ void TryInsertBounceTypesBeforeRcVertex(uint32_t data, bool doInsert)
    {
        flags &= doInsert ? ~(0xf00000) : 0xffffffff;
        flags |= doInsert ? (data << 20) : 0;
    }

    __forceinline__ __device__ __host__  void TryInsertBounceTypesAfterRcVertex(uint32_t data, bool doInsert)
    {
        flags &= doInsert ? ~(0xf000000) : 0xffffffff;
        flags |= doInsert ? (data << 24) : 0;
    }

    __forceinline__ __device__ __host__ void TransferDeltaTransmissionBSDFEvent(RestirPathFlags other, bool beforeRcVertex)
    {
        flags &= (beforeRcVertex ? ~(0xf00000) : ~(0xf000000));
        flags |= other.flags & (beforeRcVertex ? 0xf00000 : 0xf000000);//  << (beforeRcVertex ? 20 : 24);
    }

    __forceinline__ __device__ __host__ void TransferDeltaTransmissionBSDFEvent(uint32_t pathFlagData, bool beforeRcVertex)
    {
        flags &= (beforeRcVertex ? ~(0xf00000) : ~(0xf000000));
        flags |= pathFlagData & (beforeRcVertex ? 0xf00000 : 0xf000000); //  << (beforeRcVertex ? 20 : 24);
    }

    __forceinline__ __device__ __host__ void TransferDeltaTransmissionBSDFEvent(LastVertexState lvs, bool beforeRcVertex)
    {
        flags &= (beforeRcVertex ? ~(0xf00000) : ~(0xf000000));
        flags |= (lvs.data << (beforeRcVertex ? 20 : 24)) & (beforeRcVertex ? 0xf00000 : 0xf000000); //  << (beforeRcVertex ? 20 : 24);
    }

    __forceinline__ __device__ __host__ void InsertDeltaTransmissionBSDFEvent(bool isDeltaEvent, bool isTransmissionEvent, uint32_t component, bool beforeRcVertex)
    {
        uint32_t data = uint32_t(isDeltaEvent) | uint32_t(isTransmissionEvent) << 1 | component << 2;
        flags &= (beforeRcVertex ? ~(0xf00000) : ~(0xf000000));
        flags |= data << (beforeRcVertex ? 20 : 24);
    }

    __forceinline__ __device__ __host__ void InsertIsDeltaEvent(bool isDeltaEvent, bool beforeRcVertex)
    {
        flags &= (beforeRcVertex ? ~(0x100000) : ~(0x1000000));
        flags |= uint32_t(isDeltaEvent) << (beforeRcVertex ? 20 : 24);
    }

    __forceinline__ __device__ __host__ void InsertIsTransmissionEvent(bool isTransmissionEvent, bool beforeRcVertex)
    {
        flags &= (beforeRcVertex ? ~(0x200000) : ~(0x2000000));
        flags |= uint32_t(isTransmissionEvent) << (beforeRcVertex ? 21 : 25);
    }

    // component uses 2 bits
    __forceinline__ __device__ __host__ void InsertBSDFComponentType(uint32_t component, bool beforeRcVertex)
    {
        flags &= (beforeRcVertex ? ~(0xc00000) : ~(0xc000000));
        flags |= component << (beforeRcVertex ? 22 : 26);
    }

    __forceinline__ __device__ __host__ bool DecodeIsDeltaEvent(bool beforeRcVertex) const
    {
        return (flags >> (beforeRcVertex ? 20 : 24)) & 1;
    }

    __forceinline__ __device__ __host__ bool DecodeIsTransmissionEvent(bool beforeRcVertex) const
    {
        return (flags >> (beforeRcVertex ? 21 : 25)) & 1;
    }

    __forceinline__ __device__ __host__ uint32_t DecodeBSDFComponent(bool beforeRcVertex) const
    {
        return (flags >> (beforeRcVertex ? 22 : 26)) & 3;
    }

    // maximum length: 15
    __forceinline__ __device__ __host__ void InsertPathLength(int pathLength)
    {
        flags &= ~0xF;
        flags |= pathLength & 0xF;
    }

    // maximum length: 15
    __forceinline__ __device__ __host__ void InsertRcVertexLength(int rcVertexLength)
    {
        flags &= ~0xF0;
        flags |= (rcVertexLength & 0xF) << 4;
    }

    // __forceinline__ __device__ __host__ 
    // void insertSuffixRcVertexLength(int suffixRcVertexLength)
    // {
    //     flags &= ~0xF00;
    //     flags |= (suffixRcVertexLength & 0xF) << 8;
    // }

    __forceinline__ __device__ __host__ void InsertPathTreeLength(int length)
    {
        flags &= ~0xF000;
        flags |= (length & 0xF) << 12;
    }

    __forceinline__ __device__ __host__ void InsertPrefixLength(int length)
    {
        flags &= 0x0FFFFFFF;
        flags |= (length & 0xF) << 28;
    }

    __forceinline__ __device__ __host__ int PathLength() const
    {
        return flags & 0xF;
    }

    __forceinline__ __device__ __host__ int PrefixLength() const
    {
        return (flags >> 28) & 0xF;
    }

    __forceinline__ __device__ __host__ int ReconVertexLength() const
    {
        return (flags >> 4) & 0xF;
    }

    __forceinline__ __device__ __host__ int PathTreeLength() const
    {
        return (flags >> 12) & 0xF;
    }

    __forceinline__ __device__ __host__ void InsertLastVertexNEE(bool isNEE)
    {
        flags &= ~0x10000;
        flags |= (int(isNEE) & 1) << 16;
    }

    __forceinline__ __device__ __host__ bool LastVertexNEE() const
    {
        return (flags >> 16) & 1;
    }

    // reserve bit 16
    __forceinline__ __device__ __host__ void InsertLightType(uint32_t lightType)
    {
        flags &= ~0xc0000;
        flags |= ((int(lightType) & 3) << 18);
    }

    __forceinline__ __device__ __host__ uint32_t LightType() const
    {
        return (flags >> 18) & 3;
    }

    __forceinline__ __device__ __host__ void InsertUserFlag(bool val)
    {
        flags &= ~0x20000;
        flags |= (int(val) & 1) << 17;
    }

    __forceinline__ __device__ __host__ bool IsUserFlagSet() const
    {
        return (flags >> 17) & 1;
    }
};

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
