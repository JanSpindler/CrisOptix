#pragma once

#include <cuda_runtime.h>
#include <graph/LaunchParams.h>
#include <util/random.h>
#include <util/pixel_index.h>

static __forceinline__ __device__ void PrefixRetrace(
	const size_t pixelIdx,
	const glm::uvec2& prevPixelCoord,
    const PathReservoir& centralRes,
	const SurfaceInteraction& primaryInteraction,
	PCG32& rng,
	const LaunchParams& params)
{
	// Exit if primary interaction invalid
	if (!primaryInteraction.valid) { return; }

	// Exit if prev pixel is not on screen
	if (!IsPixelValid(prevPixelCoord, params)) { return; }
	const size_t prevPixelIdx = GetPixelIdx(prevPixelCoord, params);

	//
	const SurfaceInteraction& prevInteraction = params.restir.primaryInteractions[prevPixelIdx];
	if (!prevInteraction.valid) { return; }

	//
	const PathReservoir& prevRes = params.restir.pathReservoirs[prevPixelIdx];
	static constexpr int startReplayPrefixLength = 1;

    // TODO
    return;

    //if (centralRes.pathFlags.PrefixLength() > startReplayPrefixLength &&
    //    centralRes.pathFlags.PathTreeLength() >= centralRes.pathFlags.PrefixLength())
    //{
    //    HitInfo rcPrevHit = HitInfo();
    //    float3 rcPrevWo = float3(0.f);
    //    float3 thp = 1.f;
    //    float dummy;

    //    ReSTIRPathFlags tempFlag = centralReservoir.pathFlags;
    //    tempFlag.insertRcVertexLength(tempFlag.prefixLength());
    //    tempFlag.insertPathLength(restir.kMaximumRcLength);
    //    tempFlag.insertBSDFComponentType(prefixReservoirs[centralOffset].componentType(), true);

    //    thp = TraceReplayedPath(temporalPrimaryHit,

    //        temporalPrimarySd,

    //        tempFlag,
    //        centralReservoir.initRandomSeed, centralReservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, true, dummy, false, true);

    //    rcBufferOffsets[2 * centralOffset] = 2 * centralOffset;
    //    reconnectionDataBuffer[2 * centralOffset] = ReconnectionData(rcPrevHit, rcPrevWo, thp);
    //}

    //if (temporalReservoir.pathFlags.prefixLength() > startReplayPrefixLength &&
    //    temporalReservoir.pathFlags.pathTreeLength() >= temporalReservoir.pathFlags.prefixLength())
    //{
    //    HitInfo rcPrevHit = HitInfo();
    //    float3 rcPrevWo = float3(0.f);
    //    float3 thp = 1.f;
    //    float prefixPartTotalLength = 0.f;

    //    ReSTIRPathFlags tempFlag = temporalReservoir.pathFlags;
    //    tempFlag.insertRcVertexLength(tempFlag.prefixLength());
    //    tempFlag.insertPathLength(restir.kMaximumRcLength);
    //    tempFlag.insertBSDFComponentType(prevPrefixReservoirs[prevOffset].componentType(), true);

    //    thp = TraceReplayedPath(centralPrimaryHit,

    //        centralPrimarySd,

    //        tempFlag,
    //        temporalReservoir.initRandomSeed, temporalReservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, false, prefixPartTotalLength, false, true);
    //    const int pixelId = pixel.y * params.frameDim.x + pixel.x;

    //    if (restir.subpathSettings.knnSearchAdaptiveRadiusType == (uint)ConditionalReSTIR::KNNAdaptiveRadiusType::RayCone)
    //    {
    //        float prefixTotalLength = length(gScene.camera.data.posW - centralPrimarySd.posW);
    //        prefixTotalLengthBuffer[pixelId] = prefixTotalLength + prefixPartTotalLength;
    //    }
    //    rcBufferOffsets[2 * centralOffset + 1] = 2 * centralOffset + 1;
    //    reconnectionDataBuffer[2 * centralOffset + 1] = ReconnectionData(rcPrevHit, rcPrevWo, thp);
    //}
}
