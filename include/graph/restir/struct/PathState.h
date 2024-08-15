#pragma once

#include <glm/glm.hpp>
#include <graph/Interaction.h>
#include <util/random.h>
#include <graph/restir/struct/ConditionalRestirData.h>
#include <graph/restir/struct/PathReplayInfo.h>

static constexpr uint32_t kMaxRejectedHits = 16; // Maximum number of rejected hits along a path. The path is terminated if the limit is reached to avoid getting stuck in pathological cases.

static constexpr float kRayTMax = 1e30f;

// Be careful with changing these. PathFlags share 32-bit uint with vertexIndex. For now, we keep 10 bits for vertexIndex.
// PathFlags take higher bits, VertexIndex takes lower bits.
static constexpr uint32_t kVertexIndexBitCount = 10u;
static constexpr uint32_t kVertexIndexBitMask = (1u << kVertexIndexBitCount) - 1u;
static constexpr uint32_t kPathFlagsBitCount = 32u - kVertexIndexBitCount;
static constexpr uint32_t kPathFlagsBitMask = ((1u << kPathFlagsBitCount) - 1u) << kVertexIndexBitCount;
static constexpr uint32_t kBSDFComponentBitOffset = kVertexIndexBitCount + 16u; // 0x10000

enum class PathFlags
{
    active = 0x0001, ///< Path is active/terminated.
    hit = 0x0002,    ///< Result of the scatter ray (0 = miss, 1 = hit).

    transmission = 0x0004, ///< Scatter ray went through a transmission event.
    specular = 0x0008,     ///< Scatter ray went through a specular event.
    delta = 0x0010,        ///< Scatter ray went through a delta event.

    insideDielectricVolume = 0x0020, ///< Path vertex is inside a dielectric volume.
    lightSampledUpper = 0x0040,      ///< Last path vertex sampled lights using NEE (in upper hemisphere).
    lightSampledLower = 0x0080,      ///< Last path vertex sampled lights using NEE (in lower hemisphere).

    diffusePrimaryHit = 0x0100,         ///< Scatter ray went through a diffuse event on primary hit.
    specularPrimaryHit = 0x0200,        ///< Scatter ray went through a specular event on primary hit.
    deltaReflectionPrimaryHit = 0x0400, ///< Primary hit was sampled as the delta reflection.
    deltaTransmissionPath = 0x0800,     ///< Path started with and followed delta transmission events (whenever possible - TIR could be an exception) until it hit the first non-delta event.
    deltaOnlyPath = 0x1000,             ///< There was no non-delta events along the path so far.

    isPrevFrame = 0x2000,
    // kUseReSTIR
    rough = 0x4000, ///< classified as a rough path vertex for ReSTIR

    ///< for ReSTIR use (for multi-component BSDF like StandardBSDF, this tells us which component is sampled)
    component1 = 0x10000, ///< e.g. component 0: DiffuseReflection component 1: DiffuseTransmission component 2: SpecularReflection
    component2 = 0x20000, ///< Component 3: SpecularReflectionTransmission
    component3 = 0x30000, /// to know the component, bit 16-17 of the path flags is checked

    nonZeroTemporalUpdate = 0x40000,

    isTraceNewSuffix = 0x80000,
    isPrefixReplay = 0x100000,
};

struct PathState
{
    uint32_t id;                     ///< Path ID encodes (pixel, sampleIdx) with 12 bits each for pixel x|y and 8 bits for sample index.

    uint32_t flagsAndVertexIndex;    ///< Higher kPathFlagsBitCount bits: Flags indicating the current status. This can be multiple PathFlags flags OR'ed together.
    ///< Lower kVertexIndexBitCount bits: Current vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).
    uint16_t rejectedHits;           ///< Number of false intersections rejected along the path. This is used as a safeguard to avoid deadlock in pathological cases.
    float sceneLength;            ///< Path length in scene units (0.f at primary hit).
    uint32_t bounceCounters;         ///< Packed counters for different types of bounces (see BounceType).

    // Scatter ray
    glm::vec3 origin;                 ///< Origin of the scatter ray.
    glm::vec3 dir;                    ///< Scatter ray normalized direction.
    float       pdf;                    ///< Pdf for generating the scatter ray.
    glm::vec3 normal;                 ///< Shading normal at the scatter ray origin.
    SurfaceInteraction hit;

    glm::vec3 throughput;                    ///< Path throughput.
    glm::vec3 L;                      ///< Accumulated path contribution.

    //InteriorList interiorList;          ///< Interior list. Keeping track of a stack of materials with medium properties.
    PCG32 rng;
    PathReplayInfo pathReplayInfo; ///< for ReSTIR
    ConditionalRestirData restirData;     ///< ReSTIR data (used for suffix re-generation)
    float pathTotalLength;

    // Accessors
    __forceinline__ __device__ bool isTerminated() const { return !isActive(); }
    __forceinline__ __device__ bool isActive() const { return hasFlag(PathFlags::active); }
    __forceinline__ __device__ bool isHit() const { return hasFlag(PathFlags::hit); }
    __forceinline__ __device__ bool isTransmission() const { return hasFlag(PathFlags::transmission); }
    __forceinline__ __device__ bool isSpecular() const { return hasFlag(PathFlags::specular); }
    __forceinline__ __device__ bool isDelta() const { return hasFlag(PathFlags::delta); }
    __forceinline__ __device__ bool isInsideDielectricVolume() const { return hasFlag(PathFlags::insideDielectricVolume); }

    __forceinline__ __device__ bool isRough() const { return hasFlag(PathFlags::rough); }
    // TODO: generalize to arbitrary lobe types
    __forceinline__ __device__ uint32_t getSampledBSDFComponent() const { return (flagsAndVertexIndex >> kBSDFComponentBitOffset) & 3; }
    __forceinline__ __device__ bool isPrevFrame() const { return hasFlag(PathFlags::isPrevFrame); }
    __forceinline__ __device__ bool hasNonZeroTemporalUpdate() const { return hasFlag(PathFlags::nonZeroTemporalUpdate); }
    __forceinline__ __device__ bool isTraceNewSuffix() const { return hasFlag(PathFlags::isTraceNewSuffix); }
    __forceinline__ __device__ bool isPrefixReplay() const { return hasFlag(PathFlags::isPrefixReplay); }

    __forceinline__ __device__ bool isLightSampled() const
    {
        const uint32_t bits = (uint32_t(PathFlags::lightSampledUpper) | uint32_t(PathFlags::lightSampledLower)) << kVertexIndexBitCount;
        return flagsAndVertexIndex & bits;
    }

    __forceinline__ __device__ bool isLightSampledUpper() const { return hasFlag(PathFlags::lightSampledUpper); }
    __forceinline__ __device__ bool isLightSampledLower() const { return hasFlag(PathFlags::lightSampledLower); }
    __forceinline__ __device__ bool isDiffusePrimaryHit() const { return hasFlag(PathFlags::diffusePrimaryHit); }
    __forceinline__ __device__ bool isSpecularPrimaryHit() const { return hasFlag(PathFlags::specularPrimaryHit); }
    __forceinline__ __device__ bool isDeltaReflectionPrimaryHit() const { return hasFlag(PathFlags::deltaReflectionPrimaryHit); }
    __forceinline__ __device__ bool isDeltaTransmissionPath() const { return hasFlag(PathFlags::deltaTransmissionPath); }
    __forceinline__ __device__ bool isDeltaOnlyPath() const { return hasFlag(PathFlags::deltaOnlyPath); }

    // Check if the scatter event is samplable by the light sampling technique.
    __forceinline__ __device__ bool isLightSamplable() const { return !isDelta(); }

    __forceinline__ __device__ void terminate() { setFlag(PathFlags::active, false); }
    __forceinline__ __device__ void setActive() { setFlag(PathFlags::active); }
    __forceinline__ __device__ void setHit(const SurfaceInteraction& hitInfo) { hit = hitInfo; setFlag(PathFlags::hit); }
    __forceinline__ __device__ void clearHit() { setFlag(PathFlags::hit, false); }

    __forceinline__ __device__ void clearEventFlags()
    {
        const uint32_t bits = (uint32_t(PathFlags::transmission) | uint32_t(PathFlags::specular) | uint32_t(PathFlags::delta)) << kVertexIndexBitCount;
        flagsAndVertexIndex &= ~bits;
    }

    __forceinline__ __device__ void setTransmission(bool value = true) { setFlag(PathFlags::transmission, value); }
    __forceinline__ __device__ void setSpecular(bool value = true) { setFlag(PathFlags::specular, value); }
    __forceinline__ __device__ void setDelta(bool value = true) { setFlag(PathFlags::delta, value); }
    __forceinline__ __device__ void setInsideDielectricVolume(bool value = true) { setFlag(PathFlags::insideDielectricVolume, value); }
    __forceinline__ __device__ void setLightSampled(bool upper, bool lower) { setFlag(PathFlags::lightSampledUpper, upper); setFlag(PathFlags::lightSampledLower, lower); }
    __forceinline__ __device__ void setDiffusePrimaryHit(bool value = true) { setFlag(PathFlags::diffusePrimaryHit, value); }
    __forceinline__ __device__ void setSpecularPrimaryHit(bool value = true) { setFlag(PathFlags::specularPrimaryHit, value); }
    __forceinline__ __device__ void setDeltaReflectionPrimaryHit(bool value = true) { setFlag(PathFlags::deltaReflectionPrimaryHit, value); }
    __forceinline__ __device__ void setDeltaTransmissionPath(bool value = true) { setFlag(PathFlags::deltaTransmissionPath, value); }
    __forceinline__ __device__ void setDeltaOnlyPath(bool value = true) { setFlag(PathFlags::deltaOnlyPath, value); }
    __forceinline__ __device__ void setRough(bool value = true) { setFlag(PathFlags::rough, value); }
    __forceinline__ __device__ void setSampledBSDFComponent(uint32_t component)
    {
        flagsAndVertexIndex &= ~(3 << kBSDFComponentBitOffset);
        flagsAndVertexIndex |= component << kBSDFComponentBitOffset;
    }
    __forceinline__ __device__ void setPrevFrame(bool value = true) { setFlag(PathFlags::isPrevFrame, value); }
    __forceinline__ __device__ void setNonZeroTemporalUpdate(bool value = true) { setFlag(PathFlags::nonZeroTemporalUpdate, value); }
    __forceinline__ __device__ void setTraceNewSuffix(bool value = true) { setFlag(PathFlags::isTraceNewSuffix, value); }
    __forceinline__ __device__ void setPrefixReplay(bool value = true) { setFlag(PathFlags::isPrefixReplay, value); }

    __forceinline__ __device__ bool hasFlag(PathFlags flag) const
    {
        const uint32_t bit = uint32_t(flag) << kVertexIndexBitCount;
        return (flagsAndVertexIndex & bit) != 0;
    }

    __forceinline__ __device__ void setFlag(PathFlags flag, bool value = true)
    {
        const uint32_t bit = uint32_t(flag) << kVertexIndexBitCount;
        flagsAndVertexIndex = (value) ? (flagsAndVertexIndex | bit) : (flagsAndVertexIndex & ~bit);
    }

    //uint getBounces(BounceType type)
    //{
    //    const uint shift = (uint)type << 3;
    //    return (bounceCounters >> shift) & 0xff;
    //}

    //void setBounces(BounceType type, uint bounces)
    //{
    //    const uint shift = (uint)type << 3;
    //    bounceCounters = (bounceCounters & ~((uint)0xff << shift)) | ((bounces & 0xff) << shift);
    //}

    //void incrementBounces(BounceType type)
    //{
    //    const uint shift = (uint)type << 3;
    //    // We assume that bounce counters cannot overflow.
    //    bounceCounters += (1 << shift);
    //}

    __forceinline__ __device__ glm::uvec2 getPixel() const { return glm::uvec2(id & 0xFFF, (id >> 12) & 0xFFF); }
    __forceinline__ __device__ uint32_t getSampleIdx() const { return id >> 24; }

    // Unsafe - assumes that index is small enough.
    __forceinline__ __device__ void setVertexIndex(uint32_t index)
    {
        // Clear old vertex index.
        flagsAndVertexIndex &= kPathFlagsBitMask;
        // Set new vertex index (unsafe).
        flagsAndVertexIndex |= index;
    }

    __forceinline__ __device__ uint32_t getVertexIndex() const { return flagsAndVertexIndex & kVertexIndexBitMask; }

    // vertex length is vertex index - 1
    __forceinline__ __device__ int getVertexLength() const { return getVertexIndex() - 1; }

    // Unsafe - assumes that vertex index never overflows.
    __forceinline__ __device__ void incrementVertexIndex() { flagsAndVertexIndex += 1; }
    // Unsafe - assumes that vertex index will never be decremented below zero.
    __forceinline__ __device__ void decrementVertexIndex() { flagsAndVertexIndex -= 1; }

    //Ray getScatterRay()
    //{
    //    return Ray(origin, dir, 0.f, kRayTMax);
    //}
};
