#pragma once

#include <glm/glm.hpp>
#include <vector_types.h>
#include <vector_functions.h>

template <int Dim, typename Scalar>
struct cuda_type
{
    static_assert(sizeof(Scalar) == 0, "Not a cuda type (i.e. int3 or float4)!");
};

template<> struct cuda_type<1, glm::i8> { typedef char1 vector_type; typedef char scalar_type; };
template<> struct cuda_type<2, glm::i8> { typedef char2 vector_type; typedef char scalar_type; };
template<> struct cuda_type<3, glm::i8> { typedef char3 vector_type; typedef char scalar_type; };
template<> struct cuda_type<4, glm::i8> { typedef char4 vector_type; typedef char scalar_type; };
template<> struct cuda_type<1, glm::u8> { typedef uchar1 vector_type; typedef unsigned char scalar_type; };
template<> struct cuda_type<2, glm::u8> { typedef uchar2 vector_type; typedef unsigned char scalar_type; };
template<> struct cuda_type<3, glm::u8> { typedef uchar3 vector_type; typedef unsigned char scalar_type; };
template<> struct cuda_type<4, glm::u8> { typedef uchar4 vector_type; typedef unsigned char scalar_type; };
template<> struct cuda_type<1, glm::i16> { typedef short1 vector_type; typedef short scalar_type; };
template<> struct cuda_type<2, glm::i16> { typedef short2 vector_type; typedef short scalar_type; };
template<> struct cuda_type<3, glm::i16> { typedef short3 vector_type; typedef short scalar_type; };
template<> struct cuda_type<4, glm::i16> { typedef short4 vector_type; typedef short scalar_type; };
template<> struct cuda_type<1, glm::u16> { typedef ushort1 vector_type; typedef unsigned short scalar_type; };
template<> struct cuda_type<2, glm::u16> { typedef ushort2 vector_type; typedef unsigned short scalar_type; };
template<> struct cuda_type<3, glm::u16> { typedef ushort3 vector_type; typedef unsigned short scalar_type; };
template<> struct cuda_type<4, glm::u16> { typedef ushort4 vector_type; typedef unsigned short scalar_type; };
template<> struct cuda_type<1, glm::i32> { typedef int1 vector_type; typedef int scalar_type; };
template<> struct cuda_type<2, glm::i32> { typedef int2 vector_type; typedef int scalar_type; };
template<> struct cuda_type<3, glm::i32> { typedef int3 vector_type; typedef int scalar_type; };
template<> struct cuda_type<4, glm::i32> { typedef int4 vector_type; typedef int scalar_type; };
template<> struct cuda_type<1, glm::u32> { typedef uint1 vector_type; typedef unsigned int scalar_type; };
template<> struct cuda_type<2, glm::u32> { typedef uint2 vector_type; typedef unsigned int scalar_type; };
template<> struct cuda_type<3, glm::u32> { typedef uint3 vector_type; typedef unsigned int scalar_type; };
template<> struct cuda_type<4, glm::u32> { typedef uint4 vector_type; typedef unsigned int scalar_type; };
template<> struct cuda_type<1, glm::i64> { typedef longlong1 vector_type; typedef long long scalar_type; };
template<> struct cuda_type<2, glm::i64> { typedef longlong2 vector_type; typedef long long scalar_type; };
template<> struct cuda_type<3, glm::i64> { typedef longlong3 vector_type; typedef long long scalar_type; };
template<> struct cuda_type<4, glm::i64> { typedef longlong4 vector_type; typedef long long scalar_type; };
template<> struct cuda_type<1, glm::u64> { typedef ulonglong1 vector_type; typedef unsigned long long scalar_type; };
template<> struct cuda_type<2, glm::u64> { typedef ulonglong2 vector_type; typedef unsigned long long scalar_type; };
template<> struct cuda_type<3, glm::u64> { typedef ulonglong3 vector_type; typedef unsigned long long scalar_type; };
template<> struct cuda_type<4, glm::u64> { typedef ulonglong4 vector_type; typedef unsigned long long scalar_type; };
template<> struct cuda_type<1, glm::f32> { typedef float1 vector_type; typedef float scalar_type; };
template<> struct cuda_type<2, glm::f32> { typedef float2 vector_type; typedef float scalar_type; };
template<> struct cuda_type<3, glm::f32> { typedef float3 vector_type; typedef float scalar_type; };
template<> struct cuda_type<4, glm::f32> { typedef float4 vector_type; typedef float scalar_type; };
template<> struct cuda_type<1, glm::f64> { typedef double1 vector_type; typedef double scalar_type; };
template<> struct cuda_type<2, glm::f64> { typedef double2 vector_type; typedef double scalar_type; };
template<> struct cuda_type<3, glm::f64> { typedef double3 vector_type; typedef double scalar_type; };
template<> struct cuda_type<4, glm::f64> { typedef double4 vector_type; typedef double scalar_type; };

#pragma endregion cuda_type

#pragma region make_cuda_type

template<int N, typename T>
struct make_cuda_type
{
    static_assert(sizeof(T) == 0, "Not a cuda type (i.e. int3 or float4)!");
};

template <> struct make_cuda_type<1, glm::i8> : public cuda_type<1, glm::i8> { static __host__ __device__ vector_type apply(scalar_type x) { return make_char1(x); } };
template <> struct make_cuda_type<2, glm::i8> : public cuda_type<2, glm::i8> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y) { return make_char2(x, y); } };
template <> struct make_cuda_type<3, glm::i8> : public cuda_type<3, glm::i8> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_char3(x, y, z); } };
template <> struct make_cuda_type<4, glm::i8> : public cuda_type<4, glm::i8> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_char4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::u8> : public cuda_type<1, glm::u8> { static __host__ __device__ vector_type apply(scalar_type x) { return make_uchar1(x); } };
template <> struct make_cuda_type<2, glm::u8> : public cuda_type<2, glm::u8> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y) { return make_uchar2(x, y); } };
template <> struct make_cuda_type<3, glm::u8> : public cuda_type<3, glm::u8> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_uchar3(x, y, z); } };
template <> struct make_cuda_type<4, glm::u8> : public cuda_type<4, glm::u8> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_uchar4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::i16> : public cuda_type<1, glm::i16> { static __host__ __device__ vector_type apply(scalar_type x) { return make_short1(x); } };
template <> struct make_cuda_type<2, glm::i16> : public cuda_type<2, glm::i16> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y) { return make_short2(x, y); } };
template <> struct make_cuda_type<3, glm::i16> : public cuda_type<3, glm::i16> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_short3(x, y, z); } };
template <> struct make_cuda_type<4, glm::i16> : public cuda_type<4, glm::i16> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_short4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::u16> : public cuda_type<1, glm::u16> { static __host__ __device__ vector_type apply(scalar_type x) { return make_ushort1(x); } };
template <> struct make_cuda_type<2, glm::u16> : public cuda_type<2, glm::u16> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y) { return make_ushort2(x, y); } };
template <> struct make_cuda_type<3, glm::u16> : public cuda_type<3, glm::u16> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_ushort3(x, y, z); } };
template <> struct make_cuda_type<4, glm::u16> : public cuda_type<4, glm::u16> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_ushort4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::i32> : public cuda_type<1, glm::i32> { static __host__ __device__ vector_type apply(scalar_type x) { return make_int1(x); } };
template <> struct make_cuda_type<2, glm::i32> : public cuda_type<2, glm::i32> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y) { return make_int2(x, y); } };
template <> struct make_cuda_type<3, glm::i32> : public cuda_type<3, glm::i32> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_int3(x, y, z); } };
template <> struct make_cuda_type<4, glm::i32> : public cuda_type<4, glm::i32> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_int4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::u32> : public cuda_type<1, glm::u32> { static __host__ __device__ vector_type apply(scalar_type x) { return make_uint1(x); } };
template <> struct make_cuda_type<2, glm::u32> : public cuda_type<2, glm::u32> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y) { return make_uint2(x, y); } };
template <> struct make_cuda_type<3, glm::u32> : public cuda_type<3, glm::u32> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_uint3(x, y, z); } };
template <> struct make_cuda_type<4, glm::u32> : public cuda_type<4, glm::u32> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_uint4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::i64> : public cuda_type<1, glm::i64> { static __host__ __device__ vector_type apply(scalar_type x) { return make_longlong1(x); } };
template <> struct make_cuda_type<2, glm::i64> : public cuda_type<2, glm::i64> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y) { return make_longlong2(x, y); } };
template <> struct make_cuda_type<3, glm::i64> : public cuda_type<3, glm::i64> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_longlong3(x, y, z); } };
template <> struct make_cuda_type<4, glm::i64> : public cuda_type<4, glm::i64> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_longlong4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::u64> : public cuda_type<1, glm::u64> { static __host__ __device__ vector_type apply(scalar_type x) { return make_ulonglong1(x); } };
template <> struct make_cuda_type<2, glm::u64> : public cuda_type<2, glm::u64> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y) { return make_ulonglong2(x, y); } };
template <> struct make_cuda_type<3, glm::u64> : public cuda_type<3, glm::u64> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_ulonglong3(x, y, z); } };
template <> struct make_cuda_type<4, glm::u64> : public cuda_type<4, glm::u64> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_ulonglong4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::f32> : public cuda_type<1, glm::f32> { static __host__ __device__ vector_type apply(scalar_type x) { return make_float1(x); } };
template <> struct make_cuda_type<2, glm::f32> : public cuda_type<2, glm::f32> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y) { return make_float2(x, y); } };
template <> struct make_cuda_type<3, glm::f32> : public cuda_type<3, glm::f32> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_float3(x, y, z); } };
template <> struct make_cuda_type<4, glm::f32> : public cuda_type<4, glm::f32> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_float4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::f64> : public cuda_type<1, glm::f64> { static __host__ __device__ vector_type apply(scalar_type x) { return make_double1(x); } };
template <> struct make_cuda_type<2, glm::f64> : public cuda_type<2, glm::f64> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y) { return make_double2(x, y); } };
template <> struct make_cuda_type<3, glm::f64> : public cuda_type<3, glm::f64> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_double3(x, y, z); } };
template <> struct make_cuda_type<4, glm::f64> : public cuda_type<4, glm::f64> { static __host__ __device__ vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_double4(x, y, z, w); } };

#pragma endregion make_cuda_type

#pragma region glm2cuda

template <glm::length_t N, typename T, glm::qualifier Q>
struct glm2cuda_detail
{
    static_assert(sizeof(T) == 0, "Not implemented for this type!");
};

template <typename T, glm::qualifier Q>
struct glm2cuda_detail<1, T, Q>
{
    static constexpr int N = 1;
    static __host__ __device__ typename make_cuda_type<N, T>::vector_type apply(const glm::vec<N, T, Q>& v) { return make_cuda_type<N, T>::apply(v.x); };
};

template <typename T, glm::qualifier Q>
struct glm2cuda_detail<2, T, Q>
{
    static constexpr int N = 2;
    static __host__ __device__ typename make_cuda_type<N, T>::vector_type apply(const glm::vec<N, T, Q>& v) { return make_cuda_type<N, T>::apply(v.x, v.y); };
};

template <typename T, glm::qualifier Q>
struct glm2cuda_detail<3, T, Q>
{
    static constexpr int N = 3;
    static __host__ __device__ typename make_cuda_type<N, T>::vector_type apply(const glm::vec<N, T, Q>& v) { return make_cuda_type<N, T>::apply(v.x, v.y, v.z); };
};

template <typename T, glm::qualifier Q>
struct glm2cuda_detail<4, T, Q>
{
    static constexpr int N = 4;
    static __host__ __device__ typename make_cuda_type<N, T>::vector_type apply(const glm::vec<N, T, Q>& v) { return make_cuda_type<N, T>::apply(v.x, v.y, v.z, v.w); };
};

template <glm::length_t N, typename T, glm::qualifier Q>
__host__ __device__ typename cuda_type<N, T>::vector_type glm2cuda(const glm::vec<N, T, Q>& v)
{
    return glm2cuda_detail<N, T, Q>::apply(v);
}

#pragma endregion glm2cuda

#pragma region inv_cuda_type

template <typename VecType>
struct inv_cuda_type
{
    static_assert(sizeof(VecType) == 0, "Not a cuda type!");
};

template<> struct inv_cuda_type<char1> { static constexpr int dim = 1; typedef glm::i8 scalar_type; };
template<> struct inv_cuda_type<char2> { static constexpr int dim = 2; typedef glm::i8 scalar_type; };
template<> struct inv_cuda_type<char3> { static constexpr int dim = 3; typedef glm::i8 scalar_type; };
template<> struct inv_cuda_type<char4> { static constexpr int dim = 4; typedef glm::i8 scalar_type; };
template<> struct inv_cuda_type<uchar1> { static constexpr int dim = 1; typedef glm::u8 scalar_type; };
template<> struct inv_cuda_type<uchar2> { static constexpr int dim = 2; typedef glm::u8 scalar_type; };
template<> struct inv_cuda_type<uchar3> { static constexpr int dim = 3; typedef glm::u8 scalar_type; };
template<> struct inv_cuda_type<uchar4> { static constexpr int dim = 4; typedef glm::u8 scalar_type; };
template<> struct inv_cuda_type<short1> { static constexpr int dim = 1; typedef glm::i16 scalar_type; };
template<> struct inv_cuda_type<short2> { static constexpr int dim = 2; typedef glm::i16 scalar_type; };
template<> struct inv_cuda_type<short3> { static constexpr int dim = 3; typedef glm::i16 scalar_type; };
template<> struct inv_cuda_type<short4> { static constexpr int dim = 4; typedef glm::i16 scalar_type; };
template<> struct inv_cuda_type<ushort1> { static constexpr int dim = 1; typedef glm::u16 scalar_type; };
template<> struct inv_cuda_type<ushort2> { static constexpr int dim = 2; typedef glm::u16 scalar_type; };
template<> struct inv_cuda_type<ushort3> { static constexpr int dim = 3; typedef glm::u16 scalar_type; };
template<> struct inv_cuda_type<ushort4> { static constexpr int dim = 4; typedef glm::u16 scalar_type; };
template<> struct inv_cuda_type<int1> { static constexpr int dim = 1; typedef glm::i32 scalar_type; };
template<> struct inv_cuda_type<int2> { static constexpr int dim = 2; typedef glm::i32 scalar_type; };
template<> struct inv_cuda_type<int3> { static constexpr int dim = 3; typedef glm::i32 scalar_type; };
template<> struct inv_cuda_type<int4> { static constexpr int dim = 4; typedef glm::i32 scalar_type; };
template<> struct inv_cuda_type<uint1> { static constexpr int dim = 1; typedef glm::u32 scalar_type; };
template<> struct inv_cuda_type<uint2> { static constexpr int dim = 2; typedef glm::u32 scalar_type; };
template<> struct inv_cuda_type<uint3> { static constexpr int dim = 3; typedef glm::u32 scalar_type; };
template<> struct inv_cuda_type<uint4> { static constexpr int dim = 4; typedef glm::u32 scalar_type; };
template<> struct inv_cuda_type<longlong1> { static constexpr int dim = 1; typedef glm::i64 scalar_type; };
template<> struct inv_cuda_type<longlong2> { static constexpr int dim = 2; typedef glm::i64 scalar_type; };
template<> struct inv_cuda_type<longlong3> { static constexpr int dim = 3; typedef glm::i64 scalar_type; };
template<> struct inv_cuda_type<longlong4> { static constexpr int dim = 4; typedef glm::i64 scalar_type; };
template<> struct inv_cuda_type<ulonglong1> { static constexpr int dim = 1; typedef glm::u64 scalar_type; };
template<> struct inv_cuda_type<ulonglong2> { static constexpr int dim = 2; typedef glm::u64 scalar_type; };
template<> struct inv_cuda_type<ulonglong3> { static constexpr int dim = 3; typedef glm::u64 scalar_type; };
template<> struct inv_cuda_type<ulonglong4> { static constexpr int dim = 4; typedef glm::u64 scalar_type; };
template<> struct inv_cuda_type<float1> { static constexpr int dim = 1; typedef glm::f32 scalar_type; };
template<> struct inv_cuda_type<float2> { static constexpr int dim = 2; typedef glm::f32 scalar_type; };
template<> struct inv_cuda_type<float3> { static constexpr int dim = 3; typedef glm::f32 scalar_type; };
template<> struct inv_cuda_type<float4> { static constexpr int dim = 4; typedef glm::f32 scalar_type; };
template<> struct inv_cuda_type<double1> { static constexpr int dim = 1; typedef glm::f64 scalar_type; };
template<> struct inv_cuda_type<double2> { static constexpr int dim = 2; typedef glm::f64 scalar_type; };
template<> struct inv_cuda_type<double3> { static constexpr int dim = 3; typedef glm::f64 scalar_type; };
template<> struct inv_cuda_type<double4> { static constexpr int dim = 4; typedef glm::f64 scalar_type; };

template<> struct inv_cuda_type<dim3> { static constexpr int dim = 3; typedef glm::u32 scalar_type; };

#pragma endregion inv_cuda_type

#pragma region cuda2glm

template <int N, typename T, glm::qualifier Q>
struct cuda2glm_detail
{
    static_assert(sizeof(T) == 0, "Not implemented for type!");
};

template <typename T, glm::qualifier Q>
struct cuda2glm_detail<1, T, Q>
{
    typedef glm::vec<1, T, Q> result_type;
    static __host__ __device__ result_type apply(const typename cuda_type<1, T>::vector_type& v) { return result_type(v.x); }
};

template <typename T, glm::qualifier Q>
struct cuda2glm_detail<2, T, Q>
{
    typedef glm::vec<2, T, Q> result_type;
    static __host__ __device__ result_type apply(const typename cuda_type<2, T>::vector_type& v) { return result_type(v.x, v.y); }
};

template <typename T, glm::qualifier Q>
struct cuda2glm_detail<3, T, Q>
{
    typedef glm::vec<3, T, Q> result_type;
    static __host__ __device__ result_type apply(const typename cuda_type<3, T>::vector_type& v) { return result_type(v.x, v.y, v.z); }
};

template <typename T, glm::qualifier Q>
struct cuda2glm_detail<4, T, Q>
{
    typedef glm::vec<4, T, Q> result_type;
    static __host__ __device__ result_type apply(const typename cuda_type<4, T>::vector_type& v) { return result_type(v.x, v.y, v.z, v.w); }
};

template <typename CudaVectorType, glm::qualifier Q = glm::defaultp>
__host__ __device__ typename cuda2glm_detail<inv_cuda_type<CudaVectorType>::dim, typename inv_cuda_type<CudaVectorType>::scalar_type, Q>::result_type cuda2glm(const CudaVectorType& v)
{
    return cuda2glm_detail<inv_cuda_type<CudaVectorType>::dim, typename inv_cuda_type<CudaVectorType>::scalar_type, Q>::apply(v);
};
