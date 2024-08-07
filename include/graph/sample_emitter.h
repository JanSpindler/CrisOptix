#pragma once

static constexpr __device__ EmitterSample SampleEmitter(const glm::vec3& currentPos, PCG32& rng, const LaunchParams& params)
{
	// Sample emitter
	const size_t emitterCount = params.emitterTable.count;
	const size_t emitterIdx = rng.NextUint64() % emitterCount;
	const EmitterData& emitter = params.emitterTable[emitterIdx];

	// Sample emitter point
	return emitter.SamplePoint(rng);
}
