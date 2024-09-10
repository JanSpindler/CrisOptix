#pragma once

#include <cuda_runtime.h>
#include <graph/LaunchParams.h>
#include <graph/restir/ris_helper.h>
#include <graph/restir/path_gen.h>
#include <graph/trace.h>
#include <graph/restir/ShfitResult.h>

// Shifts current prefix into others domain and evaluates the target function "p hat from i"
static __forceinline__ __device__ glm::vec3 CalcCurrContribInOtherDomain(
	const SuffixPath& currSuffix,
	const SuffixPath& otherSuffix,
	float& jacobian, // Jacobian: Shift current into other
	const LaunchParams& params)
{
	// Default val jacobian
	jacobian = 0.0f;

	// Get current last prefix interaction
	const Interaction currLastPrefixInt(currSuffix.lastPrefixInt, params.transforms);
	if (!currLastPrefixInt.valid) { return glm::vec3(0.0f); }

	// Get other last perfix interaction
	Interaction otherLastPrefixInt(otherSuffix.lastPrefixInt, params.transforms);
	if (!otherLastPrefixInt.valid) { return glm::vec3(0.0f); }

	// Get reconnection interaction
	Interaction currReconInt(currSuffix.reconInt, params.transforms);
	if (!currReconInt.valid) { return glm::vec3(0.0f); }

	// Hybrid shift
	const uint32_t reconVertCount = glm::max<int>(otherSuffix.GetReconIdx() - 1, 0);
	Interaction& currInt = otherLastPrefixInt;
	PCG32 otherRng = otherSuffix.rng;
	glm::vec3 throughput(1.0f);
	for (uint32_t idx = 0; idx < reconVertCount; ++idx)
	{
		//printf("%d\n", otherSuffix.GetReconIdx());

		// Sampled brdf
		const BrdfSampleResult brdf = optixDirectCall<BrdfSampleResult, const Interaction&, PCG32&>(
			currInt.meshSbtData->sampleMaterialSbtIdx,
			currInt,
			otherRng);
		if (brdf.samplingPdf <= 0.0f) { return glm::vec3(0.0f); }
		throughput *= brdf.brdfVal;

		// Recon dir
		const glm::vec3& reconDir = brdf.outDir;

		// Trace new interaction
		const glm::vec3 oldPos = currInt.pos;
		TraceWithDataPointer<Interaction>(params.traversableHandle, oldPos, reconDir, 1e-3f, 1e16f, params.surfaceTraceParams, currInt);
		if (!currInt.valid) { return glm::vec3(0.0f); }

		// Trace occlusion
		if (TraceOcclusion(oldPos, reconDir, 1e-3f, glm::distance(oldPos, currInt.pos), params)) { return glm::vec3(0.0f); }
	}

	// Check occlusion
	const glm::vec3 reconVec = currReconInt.pos - currInt.pos;
	const float reconLen = glm::length(reconVec);
	const glm::vec3 reconDir = glm::normalize(reconVec);

	if (glm::any(glm::isinf(reconDir) || glm::isnan(reconDir))) { return glm::vec3(0.0f); }
	if (TraceOcclusion(currInt.pos, reconDir, 1e-3f, reconLen, params)) { return glm::vec3(0.0f); }

	// Brdf eval 1
	const BrdfEvalResult brdf1 = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
		currInt.meshSbtData->evalMaterialSbtIdx,
		currInt,
		reconDir);
	if (brdf1.samplingPdf <= 0.0f) { return glm::vec3(0.0f); }
	throughput *= brdf1.brdfResult;

	// Brdf eval 2
	if (currSuffix.GetReconIdx() == 0)
	{
		// Fix recon interaction in dir
		currReconInt.inRayDir = reconDir;

		// Brdf
		const BrdfEvalResult brdf2 = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
			currReconInt.meshSbtData->evalMaterialSbtIdx,
			currReconInt,
			currSuffix.reconOutDir);
		if (brdf2.samplingPdf <= 0.0f) { return glm::vec3(0.0f); }
		throughput *= brdf2.brdfResult;
	}

	// Jacobian inverse shift
	jacobian = CalcReconnectionJacobian(
		currLastPrefixInt.pos,
		currInt.pos,
		currReconInt.pos,
		currReconInt.normal);

	// Inverse shift p hat
	const glm::vec3 shiftedF = throughput * currSuffix.postReconF;

	//
	return shiftedF;
}

//static __forceinline__ __device__ ShiftResult ShiftSuffix(
//	const SuffixPath& currSuffix,
//	const SuffixPath& otherSuffix,
//	const LaunchParams& params)
//{
//	// Only reconnection shift
//	if (otherSuffix.GetReconIdx() != 1) { return ShiftResult(); }
//
//	// Trace occlusion
//	const glm::vec3 reconVec = otherSuffix.reconIntSeed.pos - currSuffix.lastPrefixIntSeed.pos;
//	const float reconLen = glm::length(reconVec);
//	const glm::vec3 reconDir = glm::normalize(reconVec);
//
//	if (glm::any(glm::isinf(reconDir) || glm::isnan(reconDir))) { return ShiftResult(); }
//	if (TraceOcclusion(otherSuffix.reconIntSeed.pos, reconDir, 1e-3f, reconLen, params)) { return ShiftResult(); }
//
//	// Get current last prefix interaction
//	Interaction currLastPrefixInt{};
//	TraceInteractionSeed(currSuffix.lastPrefixIntSeed, currLastPrefixInt, params);
//	if (!currLastPrefixInt.valid) { return ShiftResult(); }
//
//	// Brdf eval 1
//	const BrdfEvalResult brdf1 = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
//		currLastPrefixInt.meshSbtData->evalMaterialSbtIdx,
//		currLastPrefixInt,
//		reconDir);
//	if (brdf1.samplingPdf <= 0.0f) { return ShiftResult(); }
//
//	// Get other reconnection interaction
//	Interaction otherReconInt{};
//	TraceInteractionSeed(otherSuffix.reconIntSeed, otherReconInt, params);
//	if (!otherReconInt.valid) { return ShiftResult(); }
//
//	// Brdf eval 2
//	glm::vec3 brdfResult2(1.0f);
//	if (otherSuffix.GetLength() == 0)
//	{
//		const BrdfEvalResult brdf2 = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
//			otherReconInt.meshSbtData->evalMaterialSbtIdx,
//			otherReconInt,
//			otherSuffix.reconOutDir);
//		if (brdf2.samplingPdf <= 0.0f) { return ShiftResult(); }
//
//		brdfResult2 = brdf2.brdfResult;
//	}
//
//	// Shifted f
//	const glm::vec3 shiftedF = brdf1.brdfResult * brdfResult2 * otherSuffix.postReconF;
//
//	// Jacobian
//	const float jacobian = CalcReconnectionJacobian(
//		otherSuffix.lastPrefixIntSeed.pos,
//		currLastPrefixInt.pos,
//		otherReconInt.pos,
//		otherReconInt.normal);
//
//	//
//	return ShiftResult(shiftedF, jacobian);
//}
//
//static __forceinline__ __device__ void SuffixReuse(
//	Reservoir<SuffixPath>& currRes,
//	const Reservoir<SuffixPath>& otherRes,
//	const PrefixPath& prefix,
//	PCG32& rng,
//	const LaunchParams& params)
//{
//	// Assume: Both reservoirs and samples are valid
//	// Assume: Prefix is valid
//	// Assume: currRes.sample.lastPrefixPos == prefix.lastInteraction.pos
//
//	// Get prefixes from res
//	const SuffixPath& currSuffix = currRes.sample;
//	const SuffixPath& otherSuffix = otherRes.sample;
//	
//	//
//	if (!otherSuffix.IsValid()) { return; }
//
//	// Shift
//	const ShiftResult shiftResult = ShiftSuffix(currSuffix, otherSuffix, params);
//
//	// Construct shifted PrefixPath
//	const SuffixPath shiftedSuffix = SuffixPath(otherSuffix, currSuffix.lastPrefixIntSeed, shiftResult.shiftedF);
//	
//	// MIS weight | 1 = current sample and 2 = other sample | first index = domain and second index = sample
//	const float m22 = GetLuminance(otherSuffix.f) * otherRes.confidence;
//	const float m12 = CalcCurrContribInOtherDomain(currSuffix, otherSuffix, params) * currRes.confidence;
//	const float misWeight = m22 / (m12 + m22);
//
//	// Calc ris weight
//	const float risWeight = CalcResamplingWeightWi(
//		misWeight, 
//		GetLuminance(shiftResult.shiftedF), 
//		otherRes.wSum / GetLuminance(otherSuffix.f),
//		shiftResult.jacobian);
//
//	// Merge reservoirs
//	if (currRes.Merge(shiftedSuffix, otherRes.confidence, risWeight, rng))
//	{
//		// Only for debug purposes
//		//printf("%f\n", risWeight);
//	}
//}
