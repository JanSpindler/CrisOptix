#pragma once

#include <cuda_runtime.h>

struct LastVertexState
{
    uint32_t data = 0;

    __forceinline__ __device__ __host__ LastVertexState(uint32_t _data) :
        data(_data)
    {
    }

    /// WARNING: change mask in two getLastVertexState()s when adding a new field!!! 

    __forceinline__ __device__ __host__ LastVertexState(
        bool isCurrentVertexFarFromPrev,
        uint32_t lastBSDFComponent,
        bool isLastVertexDelta,
        bool isLastVertexTransmission,
        bool isLastVertexRough,
        bool isVertexClassifiedAsRoughForNEE)
        :
        data(0)
    {
        Init(
            isCurrentVertexFarFromPrev, 
            lastBSDFComponent, 
            isLastVertexDelta, 
            isLastVertexTransmission, 
            isLastVertexRough, 
            isVertexClassifiedAsRoughForNEE);
    }

    __forceinline__ __device__ __host__ void Init(
        bool isCurrentVertexFarFromPrev,
        uint32_t lastBSDFComponent,
        bool isLastVertexDelta,
        bool isLastVertexTransmission,
        bool isLastVertexRough,
        bool isVertexClassifiedAsRoughForNEE)
    {
        data = 0;
        data |= (uint32_t)isLastVertexDelta;
        data |= (uint32_t)isLastVertexTransmission << 1;
        data |= (uint32_t)lastBSDFComponent << 2;
        data |= (uint32_t)isLastVertexRough << 4;
        data |= (uint32_t)isCurrentVertexFarFromPrev << 5;
        data |= (uint32_t)isVertexClassifiedAsRoughForNEE << 6;
    }

    __forceinline__ __device__ __host__ bool IsLastVertexDelta()
    {
        return data & 1;
    }

    __forceinline__ __device__ __host__ bool IsLastVertexTransmission()
    {
        return (data >> 1) & 1;
    }

    __forceinline__ __device__ __host__ uint32_t LastBSDFComponent()
    {
        return (data >> 2) & 3;
    }

    __forceinline__ __device__ __host__ bool IsLastVertexRough()
    {
        return (data >> 4) & 1;
    }

    __forceinline__ __device__ __host__ bool IsCurrentVertexFarFromPrev()
    {
        return (data >> 5) & 1;
    }

    __forceinline__ __device__ __host__ bool IsVertexClassifiedAsRoughForNEE()
    {
        return (data >> 6) & 1;
    }
};

