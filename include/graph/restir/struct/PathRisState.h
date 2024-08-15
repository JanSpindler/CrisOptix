#pragma once

#include <graph/restir/struct/RestirPathFlags.h>
#include <glm/glm.hpp>
#include <util/random.h>
#include <graph/Interaction.h>

struct PathRisState
{
    float weight;
    glm::vec3 integrand; // the integrand value f/p in PSS
    RestirPathFlags pathFlags;
    float lightPdf; // only useful if the path ends at rcVertex
    PCG32 reconRng;

    SurfaceInteraction reconHit;

    float reconJacobian;
    glm::vec3 reconIrrad; // throughput (might only store throughput after rc)
    glm::vec3 reconInDir;

    __forceinline__ __device__ __host__ PathRisState() :
        weight(0.0f),
        integrand(0.0f),
        pathFlags(0),
        lightPdf(0.0f),
        reconRng(),
        reconHit(),
        reconJacobian(1.0f),
        reconIrrad(0.0f),
        reconInDir(0.0f)
    {
    }

//    [mutating]
//        bool addBSDFVertex(inout SampleGenerator sg, PathRcInfo pathRcInfo, uint pathLength,
//            float3 pathContribution, float3 wi, float3 Le, float3 Lr, float russianRoulettePDF, float lightPdf, uint lightType)
//    {
//        bool selected = add(sg, pathContribution, russianRoulettePDF);
//        //print(selected ? "---accepted" : "--rejected", 1111);
//        uint rcVertexLength = pathRcInfo.pathFlags.rcVertexLength();
//        bool isRcVertex = pathLength == rcVertexLength;
//        // rcVertexLength is the same as in pathRcInfo.pathFlags
//
//        this.pathFlags.flags = selected ? pathRcInfo.pathFlags.flags : this.pathFlags.flags;
//        this.pathFlags.insertPathLength(selected ? pathLength : this.pathFlags.pathLength());
//        this.pathFlags.insertLastVertexNEE(!selected && this.pathFlags.lastVertexNEE());
//        this.rcIrrad = selected ? (isRcVertex ? Le : (pathRcInfo.rcThroughput * Lr)) : this.rcIrrad;
//
//        this.rcWi = selected ? pathRcInfo.rcWi : this.rcWi;
//        if (selected) this.rcHit = pathRcInfo.rcHit;
//        this.rcJacobian = selected ? pathRcInfo.rcJacobian : this.rcJacobian;
//#if TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE
//        this.rcRandomSeed = selected ? pathRcInfo.rcRandomSeed : this.rcRandomSeed;
//#endif
//        this.integrand = selected ? pathContribution : this.integrand;
//        this.lightPdf = selected ? lightPdf : this.lightPdf;
//        this.pathFlags.insertLightType(selected ? lightType : this.pathFlags.lightType());
//
//        return selected;
//    }
//
//    [mutating]
//        bool addNeeVertex(inout SampleGenerator sg, PathRcInfo pathRcInfo, bool isRcVertex, uint pathLength,
//            float3 pathContribution, float3 wi, float3 Le, float3 Lr, float russianRoulettePDF, float lightPdf, uint lightType)
//    {
//        bool selected = add(sg, pathContribution, russianRoulettePDF);
//        uint rcVertexLength = pathRcInfo.pathFlags.rcVertexLength();
//        rcVertexLength = isRcVertex ? pathLength : rcVertexLength;
//        this.pathFlags.flags = selected ? pathRcInfo.pathFlags.flags : this.pathFlags.flags;
//        this.pathFlags.insertPathLength(selected ? pathLength : this.pathFlags.pathLength());
//        this.pathFlags.insertRcVertexLength(selected ? rcVertexLength : this.pathFlags.rcVertexLength());
//        this.pathFlags.insertLastVertexNEE(selected || this.pathFlags.lastVertexNEE());
//        this.rcIrrad = selected ? (isRcVertex ? Le : pathRcInfo.rcThroughput * Lr) : this.rcIrrad;
//        this.rcWi = selected ? (isRcVertex ? wi : pathRcInfo.rcWi) : this.rcWi;
//        // will be overwritten for isRcVertex
//        if (selected) this.rcHit = pathRcInfo.rcHit;
//        this.rcJacobian = selected ? pathRcInfo.rcJacobian : this.rcJacobian;
//#if TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE
//        this.rcRandomSeed = selected ? pathRcInfo.rcRandomSeed : this.rcRandomSeed;
//#endif        
//        this.integrand = selected ? pathContribution : this.integrand;
//        this.lightPdf = selected ? lightPdf : this.lightPdf;
//        this.pathFlags.insertLightType(selected ? lightType : this.pathFlags.lightType());
//
//        return selected;
//    }
//
//    [mutating]
//        void markAsRcVertex(uint pathLength,
//
//            HitInfo hit,
//
//            LastVertexState lastVertexState,
//            float lightPdf, uint lightType, float3 Le, float3 wi, float rcJacobian)
//    {
//        this.pathFlags.insertRcVertexLength(pathLength);
//
//        this.rcHit = hit;
//        this.pathFlags.insertIsDeltaEvent(lastVertexState.isLastVertexDelta(), true);
//        this.pathFlags.insertIsTransmissionEvent(lastVertexState.isLastVertexTransmission(), true);
//        this.pathFlags.insertBSDFComponentType(lastVertexState.lastBSDFComponent(), true);
//        this.lightPdf = lightPdf;
//
//        this.rcJacobian = rcJacobian;
//
//        this.pathFlags.insertLightType(lightType);
//        this.rcIrrad = Le;
//        this.rcWi = wi;
//    }
//
//    [mutating]
//        bool add(inout SampleGenerator sg, float3 pathContribution, float russianRoulettePDF)
//    {
//        float w = toScalar(pathContribution) / russianRoulettePDF;
//
//        w = isnan(w) ? 0.0f : w;
//        this.weight += w;
//
//        return (sampleNext1D(sg) * this.weight < w);
//    }
//
//    static float toScalar(float3 color)
//    {
//        return luminance(color); // luminance
//    }
};
