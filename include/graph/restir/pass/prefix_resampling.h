#pragma once

#include <cuda_runtime.h>
#include <graph/LaunchParams.h>
//#include <graph/restir/subpath_reuse.h>

static __forceinline__ __device__ void PrefixResampling()
{
    return;

    // TODO
    //int2 pixel = dispatchThreadId.xy;
    //const int offset = params.getReservoirOffset(pixel);
    //const int pixelId = pixel.y * params.frameDim.x + pixel.x;
    //printSetPixel(pixel);
    //logSetPixel(pixel);

    //bool insideScreen = isValidScreenRegion(params, pixel);
    //if (!insideScreen) return;

    //// in case of prefix replay, read from the rcDataBuffer
    //float3 currentReplayThp = 1.f;
    //HitInfo currentPrimaryHit;

    //ShadingData currentPrimarySd = getPixelShadingData(params, vbuffer, pixel, currentPrimaryHit);

    //if (restir.needResetTemporalHistory || !currentPrimaryHit.isValid())
    //{
    //    if (!currentPrimaryHit.isValid())
    //    {
    //        searchPointBoundingBoxBuffer[pixelId].minPoint = gScene.camera.data.posW;
    //        searchPointBoundingBoxBuffer[pixelId].maxPoint = gScene.camera.data.posW;
    //    }
    //    prefixGBuffer[offset] = restir.prefixGBuffer[offset];
    //    reservoirs[offset] = restir.pathReservoirs[offset];
    //    return;
    //}

    //bool isNeighborValid = neighborValidMask[offset].isValid(0) ||
    //    !gPathTracer.restir.subpathSettings.adaptivePrefixLength;

    //SampleGenerator sg = SampleGenerator(pixel, restir.sgCount() * params.seed + restir.prefixResamplingSgOffset());
    //const PrevCameraFrame pcf = { prevCameraU, prevCameraV, prevCameraW, prevJitterX, prevJitterY };

    //PathReservoir subpathReservoir = restir.pathReservoirs[offset];

    //int numNeighbors = restir.subpathSettings.suffixSpatialNeighborCount;
    //float2 motionVector = motionVectors[pixel];
    //int2 neighborPixel = pixel + motionVector * params.frameDim + 0.5;
    //int neighborOffset = params.getReservoirOffset(neighborPixel);
    //PrefixGBuffer outputPrefixGBuffer = restir.prefixGBuffer[offset];
    //PrefixReservoir outputPrefixReservoir = prefixReservoirs[offset];

    //// first round
    //float3 neighborReplayThp = 1.f;
    //PathReservoir neighborReservoir = prevReservoirs[neighborOffset];
    //HitInfo neighborPrimaryHit;

    //ShadingData neighborPrimarySd = getPixelTemporalShadingData(params, temporalVbuffer, neighborPixel, neighborPrimaryHit, pcf);

    //const int startReplayPrefixLength = 1;

    //if (!isValidScreenRegion(params, neighborPixel)) neighborPrimaryHit.setInvalid();

    //float pathFootprint = 0.f;

    //if (restir.subpathSettings.useMMIS)
    //{
    //    pathFootprint = outputPrefixReservoir.pathFootprint;
    //}
    //else
    //{
    //    if (gPathTracer.restir.subpathSettings.adaptivePrefixLength && subpathReservoir.pathFlags.prefixLength() > startReplayPrefixLength)
    //    {
    //        neighborReplayThp = reconnectionDataBuffer[rcBufferOffsets[2 * offset]].pathThroughput;

    //        neighborPrimarySd = loadShadingData(reconnectionDataBuffer[rcBufferOffsets[2 * offset]].rcPrevHit, float3(0.f),
    //            -reconnectionDataBuffer[rcBufferOffsets[2 * offset]].rcPrevWo, false,
    //            ExplicitLodTextureSampler(params.lodBias), true);

    //    }

    //    float prefixTotalLengthBeforeRc = 0.f;
    //    if (gPathTracer.restir.subpathSettings.adaptivePrefixLength && neighborReservoir.pathFlags.prefixLength() > startReplayPrefixLength)
    //    {
    //        currentReplayThp = reconnectionDataBuffer[rcBufferOffsets[2 * offset + 1]].pathThroughput;

    //        currentPrimarySd = loadShadingData(reconnectionDataBuffer[rcBufferOffsets[2 * offset + 1]].rcPrevHit, float3(0.f),
    //            -reconnectionDataBuffer[rcBufferOffsets[2 * offset + 1]].rcPrevWo, false,
    //            ExplicitLodTextureSampler(params.lodBias), false);

    //        prefixTotalLengthBeforeRc = prefixTotalLengthBuffer[pixelId];
    //    }
    //    else
    //    {
    //        prefixTotalLengthBeforeRc = length(currentPrimarySd.posW - gScene.camera.data.posW);
    //    }

    //    pathFootprint = prefixTotalLengthBeforeRc;

    //    ResamplePrefix(isNeighborValid, subpathReservoir,
    //        neighborReservoir, restir.subpathSettings.temporalHistoryLength,
    //        currentPrimaryHit, currentPrimarySd, neighborPrimaryHit, neighborPrimarySd,
    //        currentReplayThp, neighborReplayThp,
    //        outputPrefixGBuffer, outputPrefixReservoir,
    //        prevPrefixGBuffer[neighborOffset],
    //        prevPrefixReservoirs[neighborOffset],
    //        sg, (kMaxSurfaceBounces + 1) * offset, pathFootprint);
    //    prefixReservoirs[offset] = outputPrefixReservoir;
    //}

    //prefixGBuffer[offset] = outputPrefixGBuffer;
    //reservoirs[offset] = subpathReservoir;

    //uint pointIndex = pixelId;

    //float searchRadius;

    //if (gPathTracer.restir.subpathSettings.knnSearchAdaptiveRadiusType != (uint)ConditionalReSTIR::KNNAdaptiveRadiusType::NonAdaptive)
    //{
    //    float minRes = min(params.frameDim.x, params.frameDim.y);
    //    searchRadius = restir.subpathSettings.knnSearchRadiusMultiplier * pathFootprint * screenSpacePixelSpreadAngle;// / (minRes * minRes);
    //}
    //else
    //{
    //    // mitsuba's photon mapping initial guess
    //    searchRadius = restir.subpathSettings.knnSearchRadiusMultiplier * min(restir.sceneRadius / params.frameDim.x, restir.sceneRadius / params.frameDim.y);
    //}


    //float3 hitPosition = outputPrefixGBuffer.hit.isValid() ? loadVertexPosition(outputPrefixGBuffer.hit) : gScene.camera.data.posW;
    //searchRadius = outputPrefixGBuffer.hit.isValid() ? searchRadius : 0.f;

    //// write bounding box buffer
    //searchPointBoundingBoxBuffer[pointIndex].minPoint = hitPosition - searchRadius;
    //searchPointBoundingBoxBuffer[pointIndex].maxPoint = hitPosition + searchRadius;

    //if (gPathTracer.restir.subpathSettings.knnIncludeDirectionSearch)
    //{
    //    bool isPrefixValid = outputPrefixGBuffer.hit.isValid();
    //    prefixSearchKeys[offset].wo = isPrefixValid ? outputPrefixGBuffer.wo : 0.f;
    //}
}
