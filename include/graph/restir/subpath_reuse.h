#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <graph/restir/PathReservoir.h>
#include <graph/LaunchParams.h>
#include <graph/Interaction.h>
#include <optix_device.h>
#include <graph/trace.h>
#include <graph/restir/PrefixGBuffer.h>
#include <graph/restir/PrefixReservoir.h>

struct ReconnectionData
{
	glm::vec3 pathThroughput;
	glm::vec3 reconPrevOutDir;
};

static __forceinline__ float CompCanonPairwiseMisWeight(
	const glm::vec3& basisPathContribAtBasis,
	const glm::vec3& basisPathContribAtNeigh,
	const float basisToNeighJacobian,
	const float pairwiseK,
	const float canonM,
	const float neighM)
{
	if (GetLuminance(basisPathContribAtBasis) <= 0.0f) { return 1.0f; }

	const float atBasisTerm = GetLuminance(basisPathContribAtBasis) * canonM;
	const float misWeightBasisPath = atBasisTerm / (atBasisTerm + GetLuminance(basisPathContribAtNeigh) * basisToNeighJacobian * neighM * pairwiseK);
	return misWeightBasisPath;
}

static __forceinline__ float CompNeighPairwiseMisWeight(
	const glm::vec3& neighPathContribAtBasis,
	const glm::vec3& neighPathContribAtNeigh,
	const float neighToBasisJacobian,
	const float pairwiseK,
	const float canonM,
	const float neighM)
{
	if (GetLuminance(neighPathContribAtNeigh) <= 0.0f) { return 0.0f; }

	const float misWeightNeighPath = GetLuminance(neighPathContribAtNeigh) * neighM /
		(GetLuminance(neighPathContribAtNeigh) * neighM + GetLuminance(neighPathContribAtBasis) * neighToBasisJacobian * canonM / pairwiseK);
	return misWeightNeighPath;
}

static __forceinline__ float CalcGeomTerm(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x1FaceN)
{
	const glm::vec3 diff = x1 - x0;
	const float d2 = glm::dot(diff, diff);
	const glm::vec3 dir = diff / d2;
	return glm::abs(glm::dot(dir, x1FaceN)) / d2;
}

static __forceinline__ glm::vec3 ShiftPath(
	float& jacobian,
	const PathReservoir& srcRes,
	const ReconnectionData& reconData,
	const size_t dstPrefixLen,
	const size_t dstPrefixComp,
	const bool isPrevFrame,
	const LaunchParams& params)
{
	// TODO
	return glm::vec3(0.0f);
}

static __forceinline__ float CompTalbotMisWeightTerm(
	const PathReservoir& neighRes,
	const PathReservoir& dstRes,
	const ReconnectionData& reconData,
	const LaunchParams& params)
{
	float jacobian = 1.0f;
	const glm::vec3 integrand = ShiftPath(
		jacobian,
		neighRes,
		reconData,
		dstRes.pathFlags.PrefixLength(),
		0,
		false,
		params);
	return GetLuminance(integrand) * jacobian * dstRes.M;
}

static __forceinline__ glm::vec3 IntegrateWithNeighResSampleTalbotMis(
	const PathReservoir& centralRes,
	const PathReservoir& neighRes,
	const ReconnectionData& reconData,
	const float talbotMisWeightPartialSum,
	const LaunchParams& params)
{
	float jacobian = 1.0f;

	const uint32_t neighResPrefixLen = neighRes.pathFlags.PrefixLength();
	const uint32_t neighResPrefixComp = 0; // TODO: Make enum for brdf components -> now use diffuse as default

	const glm::vec3 integrand = ShiftPath(jacobian, neighRes, reconData, centralRes.pathFlags.PrefixLength(), 0, false, params);

	const float selfWeight = GetLuminance(neighRes.integrand) * neighRes.M;
	const float neighMisWeight = selfWeight / (talbotMisWeightPartialSum + selfWeight + GetLuminance(integrand) * jacobian * centralRes.M);

	glm::vec3 contrib = integrand * jacobian * neighRes.wSum * neighMisWeight;
	if (glm::any(glm::isinf(contrib) || glm::isnan(contrib))) { contrib = glm::vec3(0.0f); }

	return contrib;
}

static constexpr bool StreamNeighPathIntoPathResTalbotMis(
	PathReservoir& res,
	const PathReservoir& centralRes,
	const PathReservoir& neighRes,
	const ReconnectionData& reconData,
	const float talbotMisWeightPartialSum,
	PCG32& rng,
	const LaunchParams& params)
{
	float jacobianShift = 1.0f;

	const uint32_t neighResPrefixLen = neighRes.pathFlags.PrefixLength();
	const uint32_t neighResPrefixComp = 0; // TODO: Make enum for brdf components -> now use diffuse as default

	const glm::vec3 integrandShift = ShiftPath(jacobianShift, neighRes, reconData, centralRes.pathFlags.PrefixLength(), 0, false, params);

	const float selfWeight = GetLuminance(neighRes.integrand) * neighRes.M;
	const float neighMisWeight = selfWeight / (talbotMisWeightPartialSum + selfWeight + GetLuminance(integrandShift) * jacobianShift * centralRes.M);

	const bool selected = res.Merge(integrandShift, jacobianShift, neighRes, rng, neighMisWeight);
	return selected;
}

static __forceinline__ glm::vec3 IntegrateWithNeighResSample(
	const PathReservoir& centralRes,
	const PathReservoir& neighRes,
	const ReconnectionData& reconDataShift,
	const ReconnectionData& reconDataBackShift,
	const float pairwiseK,
	float& canonMisWeights,
	const LaunchParams& params)
{
	float jacobianShift = 1.0f;
	float jacobianBackShift = 1.0f;

	const uint32_t neighResPrefixLen = neighRes.pathFlags.PrefixLength();
	const uint32_t neighResPrefixComp = 0; // TODO

	const glm::vec3 integrandShift = ShiftPath(jacobianShift, neighRes, reconDataShift, centralRes.pathFlags.PrefixLength(), 0, false, params);
	const glm::vec3 integrandBackShift = ShiftPath(jacobianBackShift, centralRes, reconDataBackShift, neighResPrefixLen, neighResPrefixComp, false, params);

	const float neighMisWeight = CompNeighPairwiseMisWeight(
		integrandShift,
		neighRes.integrand,
		jacobianShift,
		pairwiseK,
		centralRes.M,
		neighRes.M);

	const float canonMisWeight = CompCanonPairwiseMisWeight(
		centralRes.integrand,
		integrandBackShift,
		jacobianBackShift,
		pairwiseK,
		centralRes.M,
		neighRes.M);
	canonMisWeights += canonMisWeight;

	glm::vec3 contrib = integrandShift * jacobianShift * neighRes.wSum * neighMisWeight;
	if (glm::any(glm::isinf(contrib) || glm::isnan(contrib))) { contrib = glm::vec3(0.0f); }
	return contrib;
}

static __forceinline__ bool StreamNeighPathIntoPathRes(
	PathReservoir& res,
	const PathReservoir& centralRes,
	const PathReservoir& neighRes,
	const ReconnectionData& reconDataShift,
	const ReconnectionData& reconDataBackShift,
	const float pairwiseK,
	float& canonMisWeight,
	PCG32& rng,
	const LaunchParams& params,
	const bool backShiftToPrevFrame = false,
	const bool tempUpdateForDynamicScene = false)
{
	float jacobianShift = 1.0f;
	float jacobianBackShift = 1.0f;

	const int neighResPrefixLength = neighRes.pathFlags.PrefixLength();
	const uint32_t neighResPrefixComp = 0; // TODO

	const glm::vec3 integrandShift = ShiftPath(
		jacobianShift, 
		neighRes, 
		reconDataShift, 
		centralRes.pathFlags.PrefixLength(), 
		0, 
		false, 
		params);
	const glm::vec3 integrandBackShift = ShiftPath(
		jacobianBackShift, 
		centralRes, 
		reconDataBackShift, 
		neighResPrefixLength, 
		neighResPrefixComp, 
		backShiftToPrevFrame, 
		params);

	const float neighMisWeight = CompNeighPairwiseMisWeight(integrandShift, neighRes.integrand, jacobianShift, pairwiseK, centralRes.M, neighRes.M);
	canonMisWeight += CompCanonPairwiseMisWeight(centralRes.integrand, integrandBackShift, jacobianBackShift, pairwiseK, centralRes.M, neighRes.M);

	const bool selected = res.Merge(integrandShift, jacobianShift, neighRes, rng, neighMisWeight);
	return selected;
}

static __forceinline__ glm::vec3 ShiftPrefixRecon(
	const SurfaceInteraction& interaction,
	const Vertex& prefixVert,
	LastVertexState& lastVertState, 
	PCG32& rng, 
	float& jacobian, 
	float& srcJacobian,
	glm::vec3& newWo,
	float& newPdf,
	float& pathFootprint,
	const bool shiftToPrevFrame,
	const LaunchParams& params)
{
	// Calc connection dir and length
	const glm::vec3 connectDir = glm::normalize(prefixVert.pos - interaction.pos);
	const float connectDistance = glm::distance(prefixVert.pos, interaction.pos);

	// TODO: Component type
	const int componentType = 0;

	// Evaluate brdf
	const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const SurfaceInteraction&, const glm::vec3&>(
		interaction.meshSbtData->evalMaterialSbtIdx,
		interaction,
		connectDir);

	// Store data from brdf evaluation
	glm::vec3 fCurrent = brdfEvalResult.brdfResult;
	const float pdfCurrent = brdfEvalResult.samplingPdf;
	newPdf = pdfCurrent;
	
	// Normalize the integrand by pdf
	if (pdfCurrent > 0.0f) { fCurrent /= pdfCurrent; }

	// Exit if not suitable for reconnection
	const float roughness = brdfEvalResult.roughness;
	const bool isRoughBounce = roughness > 0.5f; // TODO: roughness threshold
	const bool isDistant = connectDistance > 1.0f; // TODO: Segment too short threshold
	if (!isRoughBounce || isDistant) { return glm::vec3(0.0f); }

	// Geometry term
	const float geomCurrent = CalcGeomTerm(interaction.pos, prefixVert.pos, prefixVert.normal);

	// Jacobian
	jacobian = geomCurrent * pdfCurrent / srcJacobian;
	srcJacobian = geomCurrent * pdfCurrent;
	if (glm::isnan(jacobian) || glm::isinf(jacobian)) { jacobian = 0.0f; }

	// New out dir
	newWo = -connectDir;
	
	// Add to path footprint
	pathFootprint += connectDistance;

	// Check if visible
	const bool occluded = TraceOcclusion(
		params.traversableHandle,
		interaction.pos,
		connectDir,
		1e-3f,
		connectDistance,
		params.occlusionTraceParams);

	// Return normalized integrand if visible
	return occluded ? glm::vec3(0.0f) : fCurrent;
}

static __forceinline__ glm::vec3 ShiftPrefixReplay(
	const SurfaceInteraction& interaction,
	LastVertexState& lastVertState,
	PCG32& rng,
	glm::vec3& newWo,
	float& newPdf,
	float& pathFootprint,
	const LaunchParams& params)
{
	// Sample brdf using rng
	const BrdfSampleResult brdfSampleResult = optixDirectCall<BrdfSampleResult, const SurfaceInteraction&, PCG32&>(
		interaction.meshSbtData->sampleMaterialSbtIdx,
		interaction,
		rng);
	if (brdfSampleResult.samplingPdf <= 0.0f) { return glm::vec3(0.0f); }

	// Store result from brdf sampling
	newPdf = brdfSampleResult.samplingPdf;
	newWo = -brdfSampleResult.outDir;

	// Check if rough bounce
	const float roughness = brdfSampleResult.roughness;
	const bool isRoughBounce = brdfSampleResult.diffuse || roughness > 0.5f; // TODO: roughness threshold

	// Sample surface interaction
	SurfaceInteraction newInteraction{};
	TraceWithDataPointer<SurfaceInteraction>(
		params.traversableHandle,
		interaction.pos,
		brdfSampleResult.outDir,
		1e-3f,
		1e16f,
		params.surfaceTraceParams,
		&newInteraction);
	if (!newInteraction.valid) { return glm::vec3(0.0f); }

	// Calc vector between v_i and v_{i+1}
	const glm::vec3 vector = newInteraction.pos - interaction.pos;
	pathFootprint += glm::length(vector);
	const bool isDistant = glm::length(vector) > 1.0f; // TODO: distance thresold

	// Exit if more suitable for reconnection shift
	if (isDistant && isRoughBounce) { return glm::vec3(0.0f); }

	// Set last vert state
	lastVertState.Init(isDistant, 0, false, false, isRoughBounce, isRoughBounce);

	// Return
	return brdfSampleResult.weight;
}

static __forceinline__ glm::vec3 ShiftPrefix(
	const SurfaceInteraction& interaction,
	const Vertex& currentPrefixVert,
	PCG32& neighInitRng,
	const uint32_t prefixLen,
	const Vertex& neighPrefixVert,
	LastVertexState& lastVertState,
	const bool needRandomReplay,
	float& jacobian,
	float& srcJacobian,
	glm::vec3& newWo,
	Vertex& prefixVert,
	float& newPdf,
	float& pathFootprint,
	const bool shiftToPrevFrame,
	const LaunchParams& params)
{
	// Default vals
	newWo = glm::vec3(0.0f);
	prefixVert = {};
	jacobian = 0.0f;
	newPdf = 0.0f;

	// First path segment
	const glm::vec3 x0x1 = currentPrefixVert.pos - params.cameraData.pos;

	// Check if should reconnect
	bool shouldReconnect = true;
	// TODO
	//if (!adaptivePrefixLength)
	//{
	//	shouldReconnect = !needRandomReplay;
	//}
	//else if (prefixLen > 2)
	//{
	//	shouldReconnect = roughness > threshold
	//}

	// TODO: Check if neighbor prefix hit is valid
	//if (!neighPrefixVert.valid)
	//{
	//	return glm::vec3(0.0f);
	//}

	// Shift
	glm::vec3 ret(0.0f);
	if (shouldReconnect)
	{
		prefixVert = neighPrefixVert;
		
		//VertexData neighborPrefixVd = loadVertexData(neighborPrefixHit, shiftToPrevFrame);
		ret = ShiftPrefixRecon(
			interaction, 
			prefixVert, 
			lastVertState, 
			neighInitRng, 
			jacobian, 
			srcJacobian, 
			newWo, 
			newPdf, 
			pathFootprint, 
			shiftToPrevFrame, 
			params);
	}
	else
	{
		//if (adaptivePrefixLength) { return glm::vec3(0.0f); }
		jacobian = 1.0f;
		// TODO: pass prefixHit.isValid
		ret = ShiftPrefixReplay(interaction, lastVertState, neighInitRng, newWo, newPdf, pathFootprint, params);
	}

	// Return
	return ret;
}

static __forceinline__ bool ResamplePrefix(
	const bool isNeighborValid,
	PathReservoir& res,
	const PathReservoir& neighRes,
	const int temporalHistoryLength,

	const SurfaceInteraction& currentInteraction,
	const SurfaceInteraction& neighInteraction,

	const glm::vec3& currentReplayThroughput,
	const glm::vec3& neighReplayThroughput,
	PrefixGBuffer& currentPrefix,
	PrefixReservoir& currentPrefixRes,
	const PrefixGBuffer& neighPrefix,
	PrefixReservoir& neighPrefixRes,
	PCG32& rng,
	int scratchVertexBufferOffset,
	float& pathFootprint,
	
	const LaunchParams& params)
{
	// Defaults and init
	float jacobianShift = 1.0f;
	glm::vec3 outDirShifted(0.0f);

	SurfaceInteraction shiftedPrefixHit{};

	float pdfShift = 0.0f;
	neighPrefixRes.setM(glm::min(temporalHistoryLength * currentPrefixRes.M(), neighPrefixRes.M()));
	const uint32_t shiftedComponentType = neighPrefixRes.componentType();

	LastVertexState shiftedLastVertState{};
	shiftedLastVertState.Init(false, shiftedComponentType, false, false, false, false);

	float neighJacobian = neighPrefixRes.reconJacobian;

	// Shift prefix
	PCG32 neighInitRngCopy = neighRes.initRng;
	glm::vec3 fShift = ShiftPrefix(
		currentInteraction,
		{},
		neighInitRngCopy,
		neighRes.pathFlags.PrefixLength(),
		neighPrefix.interaction,
		shiftedLastVertState,
		neighPrefixRes.needRandomReplay(),
		jacobianShift,
		neighJacobian,
		outDirShifted,
		shiftedPrefixHit,
		pdfShift,
		pathFootprint,
		false,
		params);
	fShift *= currentReplayThroughput;

	//
	const uint32_t backShiftedComponentType = currentPrefixRes.componentType();
	LastVertexState backShiftedLastVertState{};
	backShiftedLastVertState.Init(false, backShiftedComponentType, false, false, false, false);

	float pdfBackShift = 0.0f;
	glm::vec3 outDirBackShift(0.0f);
	float jacobianBackShift = 1.0f;
	float pathFootprintDummy = 0.0f;
	float currentJacobian = currentPrefixRes.reconJacobian;

	SurfaceInteraction backShiftedPrefixHit{};

	// Back shift
	glm::vec3 fBackShift = ShiftPrefix(
		neighInteraction,
		{},
		res.initRng,
		res.pathFlags.PrefixLength(),
		currentPrefix.interaction,
		backShiftedLastVertState,
		currentPrefixRes.needRandomReplay(),
		jacobianBackShift,
		currentJacobian,
		outDirBackShift,
		backShiftedPrefixHit,
		pdfBackShift,
		pathFootprintDummy,
		true,
		params);
	fBackShift *= neighReplayThroughput;

	// MIS weights
	const float pairwiseK = 1.0f;
	float neighMisWeight = CompNeighPairwiseMisWeight(
		fShift, 
		glm::vec3(neighPrefixRes.pHat), 
		jacobianShift, 
		pairwiseK, 
		currentPrefixRes.M(), 
		neighPrefixRes.M());
	float canonMisWeight = CompCanonPairwiseMisWeight(
		glm::vec3(currentPrefixRes.pHat),
		fBackShift,
		jacobianBackShift,
		pairwiseK,
		currentPrefixRes.M(),
		neighPrefixRes.M());

	if (!isNeighborValid)
	{
		neighMisWeight = 0.0f;
		canonMisWeight = 1.0f;
	}

	// Weight
	float weight = currentPrefixRes.pHat * currentPrefixRes.W * canonMisWeight;
	if (glm::isnan(weight) || glm::isinf(weight)) { weight = 0.0f; }
	currentPrefixRes.W = weight;

	// Stream neighbor
	weight = GetLuminance(fShift) * neighPrefixRes.W * neighMisWeight * jacobianShift;
	if (glm::isnan(weight) || glm::isinf(weight)) { weight = 0.0f; }
	currentPrefixRes.W += weight;
	currentPrefixRes.increaseM(neighPrefixRes.M());

	// Check if accepted
	const bool selectPrev = rng.NextFloat() * currentPrefixRes.W < weight;
	if (selectPrev)
	{
		currentPrefix.interaction = shiftedPrefixHit;
		currentPrefixRes.setComponentType(shiftedComponentType);
		currentPrefixRes.setNeedRandomReplay(neighPrefixRes.needRandomReplay());
		currentPrefixRes.reconJacobian = neighJacobian;
		currentPrefixRes.pHat = GetLuminance(fShift);
		currentPrefixRes.pathFootprint = pathFootprint;
		currentPrefix.outDir = outDirShifted;
		res = neighRes;
	}
	else
	{
		pathFootprint = currentPrefixRes.pathFootprint;
	}

	currentPrefixRes.W /= currentPrefixRes.pHat;
	currentPrefixRes.W = glm::isnan(currentPrefixRes.W) || glm::isinf(currentPrefixRes.W) ?
		0.0f :
		glm::max(0.0f, currentPrefixRes.W);

	res.pathFlags.InsertUserFlag(selectPrev);
	return selectPrev;
}
