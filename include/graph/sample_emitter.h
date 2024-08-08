#pragma once

static constexpr __device__ EmitterSample SampleEmitter(PCG32& rng, const CuBufferView<EmitterData>& emitterTable)
{
	// Sample emitter
	const size_t emitterCount = emitterTable.count;
	const size_t emitterIdx = rng.NextUint64() % emitterCount;
	const EmitterData& emitter = emitterTable[emitterIdx];

	// Sample emitter point
	return emitter.SamplePoint(rng);
}
