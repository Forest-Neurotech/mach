#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <helper_cuda.h>
#include <helper_math.h> // float2 lerp

#include <iostream>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

// Add RAII support for CUDA memory
#include <nanobind/stl/optional.h>
#include <thrust/allocate_unique.h>
#include <thrust/device_allocator.h>
#include <thrust/detail/raw_pointer_cast.h>

namespace nb = nanobind;
using namespace nb::literals;

/**
 * @brief Interpolation types for sensor data sampling
 */
enum class InterpolationType {
    NearestNeighbor = 0,  ///< Use nearest neighbor (no interpolation)
    Linear = 1,           ///< Use linear interpolation (default)
    Quadratic = 2         ///< Use quadratic interpolation
};

#ifdef CUDA_PROFILE
#include <chrono>
#include <string>

/** @brief Get the current timestamp in milliseconds */
double get_timestamp_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double, std::milli>(duration).count();
}

/** @brief Struct to track section timing */
struct SectionTimer {
    const char* name;
    double start_time;
    SectionTimer(const char* section_name) : name(section_name) {
        start_time = get_timestamp_ms();
        printf("TIMER_START: %s (%.3f ms)\n", name, start_time);
    }
    ~SectionTimer() {
        double end_time = get_timestamp_ms();
        double elapsed = end_time - start_time;
        printf("TIMER_END: %s (%.3f ms, elapsed: %.3f ms)\n", name, end_time, elapsed);
    }
};

// Macro that doesn't use C++11 features incompatible with CUDA
#define TIME_SECTION(name) SectionTimer section_timer(name)
#define TIME_FUNCTION() SectionTimer function_timer(__func__)

#else
// No-op implementations when profiling is disabled
#define TIME_SECTION(name)
#define TIME_FUNCTION()
#endif

#ifndef PI
#define PI 3.14159265358979323846f
#endif

// CUDA block and cache configuration
#define DEFAULT_RECEIVE_ELEMENTS_BATCH_SIZE 140  // 8960 elements / 64 batches; might be able to increase to 150
#define DEFAULT_NUM_VOXELS_PER_BLOCK 8  // Tuned for large-scale beamforming with many frames
#define VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE (DEFAULT_NUM_VOXELS_PER_BLOCK * DEFAULT_RECEIVE_ELEMENTS_BATCH_SIZE)
// Results in 140 * 8 * sizeof(float2) = 8960 bytes < 9.6kB, which is the max shared memory per block for max-occupancy

// Some extra-tuning parameters for small-frame-count beamforming (i.e. not ensembles)
// Keep VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE constant, and adjust voxels_per_block and receive_elements_batch_size
#define MAX_FRAME_THREADS_PER_BLOCK 32
#define MIN_THREADS_PER_BLOCK 32
#ifndef CACHE_CONFIG
#define CACHE_CONFIG cudaFuncCachePreferEqual
#endif

// Custom CUDA debug assertion macro that can be toggled on/off with:
// nvcc -D CUDA_DEBUG kernel.cu
// cccl also defines an assert.h header that we may want to use instead
#ifdef CUDA_DEBUG
    #define DEBUG_ASSERT(condition) assert(condition)
#else
    #define DEBUG_ASSERT(condition) ((void)0)
#endif

/**
 * @brief Convert a stored channel-data sample to the FP32 compute type.
 *
 * FP16 support changes only the STORAGE format of channel_data: samples are
 * loaded as __half/__half2 and converted to float/float2 immediately, so all
 * interpolation, apodization, phase rotation, and accumulation stay in FP32.
 */
template<typename ComputeType, typename StorageType>
__device__ __forceinline__ ComputeType to_compute(StorageType v);
template<> __device__ __forceinline__ float to_compute<float, float>(float v) { return v; }
template<> __device__ __forceinline__ float2 to_compute<float2, float2>(float2 v) { return v; }
template<> __device__ __forceinline__ float to_compute<float, __half>(__half v) { return __half2float(v); }
template<> __device__ __forceinline__ float2 to_compute<float2, __half2>(__half2 v) { return __half22float2(v); }

/**
 * @brief Template function for nearest neighbor interpolation with bounds checking
 * @tparam DataType Either float or float2
 * @tparam StorageType Storage type of channel_data: DataType, or its FP16 counterpart (__half / __half2)
 * @param channel_data Pointer to sensor data
 * @param sample_idx Floating point sample index
 * @param receive_element_idx Index of receive element
 * @param frame_idx Index of frame
 * @param n_samples Number of samples per element
 * @param frame_stride Frames allocated per sample in channel_data (its innermost stride)
 * @param[out] is_valid Whether the sample is within bounds
 * @return Interpolated sensor sample (undefined if is_valid is false)
 */
template<typename DataType, typename StorageType = DataType>
__device__ __forceinline__ DataType interpolate_nearest(
    const StorageType* const __restrict__ channel_data,
    float sample_idx,
    uint32_t receive_element_idx,
    uint32_t frame_idx,
    uint32_t n_samples,
    uint32_t frame_stride,
    bool& is_valid
) {
    // For nearest neighbor, check if rounded sample is in bounds
    if ((sample_idx < -0.5f) || (sample_idx > (n_samples - 0.5f))) {
        is_valid = false;
        return DataType{};  // Return default-constructed value (won't be used)
    }

    const unsigned int sample_idx_round = __float2uint_rn(sample_idx);  // Round to nearest
    DEBUG_ASSERT(sample_idx_round < n_samples);  // Verify sample index is in bounds
    const uint32_t channel_data_idx = receive_element_idx * n_samples * frame_stride +
                                     sample_idx_round * frame_stride +
                                     frame_idx;
    DEBUG_ASSERT(channel_data_idx < static_cast<uint64_t>(n_samples) * frame_stride * (receive_element_idx + 1));  // Verify channel data index is in bounds

    is_valid = true;
    return to_compute<DataType>(channel_data[channel_data_idx]);
}

/**
 * @brief Template function for linear interpolation with bounds checking
 * @tparam DataType Either float or float2
 * @tparam StorageType Storage type of channel_data: DataType, or its FP16 counterpart (__half / __half2)
 * @param channel_data Pointer to sensor data
 * @param sample_idx Floating point sample index
 * @param receive_element_idx Index of receive element
 * @param frame_idx Index of frame
 * @param n_samples Number of samples per element
 * @param frame_stride Frames allocated per sample in channel_data (its innermost stride)
 * @param[out] is_valid Whether the sample is within bounds
 * @return Interpolated sensor sample (undefined if is_valid is false)
 */
template<typename DataType, typename StorageType = DataType>
__device__ __forceinline__ DataType interpolate_linear(
    const StorageType* const __restrict__ channel_data,
    float sample_idx,
    uint32_t receive_element_idx,
    uint32_t frame_idx,
    uint32_t n_samples,
    uint32_t frame_stride,
    bool& is_valid
) {
    // For linear interpolation, check if floor/ceil samples are in bounds
    if ((sample_idx < 0.0f) || (sample_idx > (n_samples - 1))) {
        is_valid = false;
        return DataType{};  // Return default-constructed value (won't be used)
    }

    const unsigned int sample_idx_floor = __float2uint_rd(sample_idx);
    const unsigned int sample_idx_ceil = __float2uint_ru(sample_idx);
    const float lerp_alpha = sample_idx - (float)sample_idx_floor;

    DEBUG_ASSERT(sample_idx_floor < n_samples);  // Verify floor sample index is in bounds
    DEBUG_ASSERT(sample_idx_ceil < n_samples);   // Verify ceil sample index is in bounds

    const uint32_t channel_data_idx_floor = receive_element_idx * n_samples * frame_stride +
                                           sample_idx_floor * frame_stride +
                                           frame_idx;
    const uint32_t channel_data_idx_ceil = receive_element_idx * n_samples * frame_stride +
                                          sample_idx_ceil * frame_stride +
                                          frame_idx;

    DEBUG_ASSERT(channel_data_idx_floor < static_cast<uint64_t>(n_samples) * frame_stride * (receive_element_idx + 1));  // Verify floor channel data index is in bounds
    DEBUG_ASSERT(channel_data_idx_ceil < static_cast<uint64_t>(n_samples) * frame_stride * (receive_element_idx + 1));   // Verify ceil channel data index is in bounds

    is_valid = true;
    return lerp(to_compute<DataType>(channel_data[channel_data_idx_floor]),
                to_compute<DataType>(channel_data[channel_data_idx_ceil]), lerp_alpha);
}

/**
 * @brief Template function for quadratic interpolation with bounds checking
 * @tparam DataType Either float or float2
 * @tparam StorageType Storage type of channel_data: DataType, or its FP16 counterpart (__half / __half2)
 * @param channel_data Pointer to sensor data
 * @param sample_idx Floating point sample index
 * @param receive_element_idx Index of receive element
 * @param frame_idx Index of frame
 * @param n_samples Number of samples per element
 * @param frame_stride Frames allocated per sample in channel_data (its innermost stride)
 * @param[out] is_valid Whether the sample is within bounds
 * @return Interpolated sensor sample (undefined if is_valid is false)
 */
template<typename DataType, typename StorageType = DataType>
__device__ __forceinline__ DataType interpolate_quadratic(
    const StorageType* const __restrict__ channel_data,
    float sample_idx,
    uint32_t receive_element_idx,
    uint32_t frame_idx,
    uint32_t n_samples,
    uint32_t frame_stride,
    bool& is_valid
) {
        // For quadratic interpolation, we need 3 points centered around sample_idx
    // Check if all 3 points are in bounds
    if ((sample_idx < 1.0f) || (sample_idx > (n_samples - 2.0f))) {
        is_valid = false;
        return DataType{};  // Return default-constructed value (won't be used)
    }

    const unsigned int sample_idx_center = __float2uint_rn(sample_idx);  // Round to nearest for better symmetry

        // Indices for the 3 points: (center-1), center, (center+1)
    const unsigned int idx_neg1 = sample_idx_center - 1;  // Left point (x=-1)
    const unsigned int idx_0 = sample_idx_center;         // Center point (x=0)
    const unsigned int idx_1 = sample_idx_center + 1;     // Right point (x=1)

    DEBUG_ASSERT(idx_neg1 < n_samples);  // Verify left point is in bounds
    DEBUG_ASSERT(idx_0 < n_samples);     // Verify center point is in bounds
    DEBUG_ASSERT(idx_1 < n_samples);     // Verify right point is in bounds

    // Calculate channel data indices
    const uint32_t base_idx = receive_element_idx * n_samples * frame_stride + frame_idx;
    const uint32_t channel_data_idx_neg1 = base_idx + idx_neg1 * frame_stride;
    const uint32_t channel_data_idx_0 = base_idx + idx_0 * frame_stride;
    const uint32_t channel_data_idx_1 = base_idx + idx_1 * frame_stride;

    DEBUG_ASSERT(channel_data_idx_neg1 < static_cast<uint64_t>(n_samples) * frame_stride * (receive_element_idx + 1));
    DEBUG_ASSERT(channel_data_idx_0 < static_cast<uint64_t>(n_samples) * frame_stride * (receive_element_idx + 1));
    DEBUG_ASSERT(channel_data_idx_1 < static_cast<uint64_t>(n_samples) * frame_stride * (receive_element_idx + 1));

    // Calculate Lagrange basis weights using actual sample_idx
    // For points at (center-1), center, (center+1), evaluating at sample_idx
    const float x = sample_idx - (float)sample_idx_center;  // x relative to center point

    // Lagrange basis polynomials:
    // L₋₁(x) = x(x-1)/2     (for point at center-1)
    // L₀(x) = (1-x)(1+x)    (for point at center)
    // L₁(x) = x(x+1)/2      (for point at center+1)
    const float w_neg1 = 0.5f * x * (x - 1.0f);      // Weight for left point (x=-1)
    const float w_0 = (1.0f - x) * (1.0f + x);       // Weight for center point (x=0)
    const float w_1 = 0.5f * x * (x + 1.0f);         // Weight for right point (x=1)

    // Get the 3 data points
    const DataType data_neg1 = to_compute<DataType>(channel_data[channel_data_idx_neg1]);  // Left point (x=-1)
    const DataType data_0 = to_compute<DataType>(channel_data[channel_data_idx_0]);        // Center point (x=0)
    const DataType data_1 = to_compute<DataType>(channel_data[channel_data_idx_1]);        // Right point (x=1)

    // Compute weighted sum using Lagrange basis
    is_valid = true;
    return w_neg1 * data_neg1 + w_0 * data_0 + w_1 * data_1;
}

/**
 * @brief Unified template function for interpolation dispatch with bounds checking
 * @tparam DataType Either float or float2
 * @tparam StorageType Storage type of channel_data: DataType, or its FP16 counterpart (__half / __half2)
 * @tparam interpType Interpolation type (compile-time constant)
 * @param channel_data Pointer to sensor data
 * @param sample_idx Floating point sample index
 * @param receive_element_idx Index of receive element
 * @param frame_idx Index of frame
 * @param n_samples Number of samples per element
 * @param frame_stride Frames allocated per sample in channel_data (its innermost stride)
 * @param[out] is_valid Whether the sample is within bounds
 * @return Interpolated sensor sample (undefined if is_valid is false)
 */
template<typename DataType, InterpolationType interpType, typename StorageType = DataType>
__device__ __forceinline__ DataType interpolate_sample(
    const StorageType* const __restrict__ channel_data,
    float sample_idx,
    uint32_t receive_element_idx,
    uint32_t frame_idx,
    uint32_t n_samples,
    uint32_t frame_stride,
    bool& is_valid
) {
    if constexpr (interpType == InterpolationType::NearestNeighbor) {
        return interpolate_nearest<DataType, StorageType>(channel_data, sample_idx, receive_element_idx, frame_idx, n_samples, frame_stride, is_valid);
    } else if constexpr (interpType == InterpolationType::Linear) {
        return interpolate_linear<DataType, StorageType>(channel_data, sample_idx, receive_element_idx, frame_idx, n_samples, frame_stride, is_valid);
    } else if constexpr (interpType == InterpolationType::Quadratic) {
        return interpolate_quadratic<DataType, StorageType>(channel_data, sample_idx, receive_element_idx, frame_idx, n_samples, frame_stride, is_valid);
    }
}

/**
 * @brief Load FPT consecutive frames' complex samples with one or two 128-bit
 * vector loads and convert to FP32.
 *
 * Requires 16-byte alignment of `p`. The inverted-kernel dispatcher guarantees
 * this by checking the base pointer and by only engaging when frame_stride keeps
 * every sample row 16B-aligned, which for FPT == 4 also keeps every frame0
 * offset aligned for both storage types.
 */
template<typename StorageType, int FPT>
__device__ __forceinline__ void load_frames(const StorageType* __restrict__ p, float2 (&out)[FPT]);

template<>
__device__ __forceinline__ void load_frames<float2, 4>(const float2* __restrict__ p, float2 (&out)[4]) {
    const float4* v = reinterpret_cast<const float4*>(p);
    const float4 a = v[0];
    const float4 b = v[1];
    out[0] = make_float2(a.x, a.y);
    out[1] = make_float2(a.z, a.w);
    out[2] = make_float2(b.x, b.y);
    out[3] = make_float2(b.z, b.w);
}

template<>
__device__ __forceinline__ void load_frames<__half2, 4>(const __half2* __restrict__ p, float2 (&out)[4]) {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    out[0] = __half22float2(*reinterpret_cast<const __half2*>(&raw.x));
    out[1] = __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
    out[2] = __half22float2(*reinterpret_cast<const __half2*>(&raw.z));
    out[3] = __half22float2(*reinterpret_cast<const __half2*>(&raw.w));
}

/**
 * Calculate the number of voxels to process per block based on frame count.
 * For small frame counts (< 4), we increase voxels_per_block to maintain at least MIN_THREADS_PER_BLOCK threads.
 * For larger frame counts, we use DEFAULT_NUM_VOXELS_PER_BLOCK for optimal performance.
 *
 * @param frames_per_block Number of frames to process in this block
 * @return Number of voxels to process per block
 */
static inline __host__ __device__ int calculate_voxels_per_block(int frames_per_block) {
    // For small frame counts, increase voxels to maintain minimum thread count
    // For larger frame counts, use default voxels for optimal performance
    DEBUG_ASSERT(frames_per_block > 0);
    return max((MIN_THREADS_PER_BLOCK + frames_per_block - 1) / frames_per_block, DEFAULT_NUM_VOXELS_PER_BLOCK);
}

/**
 * Calculate the number of receive elements to process per batch based on voxels per block.
 * This maintains constant shared memory usage by scaling inversely with voxels_per_block.
 *
 * @param voxels_per_block Number of voxels being processed in this block
 * @return Number of receive elements to process per batch
 */
static inline __host__ __device__ int calculate_receive_elements_batch_size(int voxels_per_block) {
    // Scale receive elements inversely with voxels to maintain constant shared memory usage
    DEBUG_ASSERT(voxels_per_block > 0);
    return VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE / voxels_per_block;
}


/**
 * @brief Tukey window apodization function
 *
 * @param r_norm: float, the normalized distance from the center of the aperture
 * @param alpha: float, the alpha parameter for the Tukey window
 *   - Range [0, 1]:
 *   - 0.0: no apodization (rectangular window)
 *   - 0.5: moderate apodization
 *   - 1.0: maximum apodization (Hann window)
 *
 * @remark: this is technically a half-window, because we only need
 * the positive half of the window for r_norm ∈ [0, 1].
 *
 * @remark: for more flexible apodization windows, we can pass in a
 * precomputed window array to the kernel, and use texture memory to
 * look up the apodization weight for each element.
 * https://github.com/Forest-Neurotech/mach/commit/580732cfe0f837b72b56f52d0ed035770546adfb
 */
static __device__ __forceinline__ float tukey_apod_weight(float r_norm, float alpha) {
    DEBUG_ASSERT(alpha >= 0.0f && alpha <= 1.0f);
    if ((r_norm < 0.0f) || (r_norm > 1.0f)) {
        return 0.0f;
    }
    if (r_norm <= (1.0f - alpha)) {
        // flat region
        return 1.0f;
    }
    // positive-taper region
    // use the mirror of the negative-taper region
    // https://en.wikipedia.org/wiki/Window_function#Tukey_window
    // r_norm_mirror corresponds to n/2 in the negative-taper region
    const float r_norm_mirror = 1.0f - r_norm;
    const float weight = 0.5f - 0.5f * cosf(PI * r_norm_mirror / alpha);
    return weight;
}

/**
 * @brief Check CUDA driver compatibility and warn if incompatible
 *
 * This function checks if the installed CUDA driver is compatible with the
 * NVCC version used to compile this module. Issues a warning if incompatible.
 */
static void checkCudaDriverCompatibility() {
    int driverVersion = 0;

    // Get driver version - if this fails, let later CUDA operations handle the error
    if (cudaDriverGetVersion(&driverVersion) != cudaSuccess) {
        PyErr_WarnEx(PyExc_RuntimeWarning, "Could not get CUDA driver version", 1);
        return;
    }

    // Use pre-parsed NVCC version (no runtime parsing needed!)
    constexpr int nvccMajor = NVCC_MAJOR;
    constexpr int nvccMinor = NVCC_MINOR;

    int driverMajor = driverVersion / 1000;
    int driverMinor = (driverVersion / 10) % 100;

    if ((driverMajor > nvccMajor) || (driverMajor == nvccMajor && driverMinor >= nvccMinor)) {
        return;
    }

    // Python-style warning that driver is too old
    std::string warning_msg =
        "[mach] CUDA driver version (" +
        std::to_string(driverMajor) + "." +
        std::to_string(driverMinor) +
        ") is too old for code compiled with NVCC " + NVCC_VERSION_STR +
        ". You may see kernel launch failures or crashes.\n" +
        "→ Please update your NVIDIA driver to version " +
        std::to_string(nvccMajor) + "." + std::to_string(nvccMinor) + " or newer.";

    PyErr_WarnEx(PyExc_RuntimeWarning, warning_msg.c_str(), 1);
}

/**
 * @brief Check compute capability and warn if too old
 *
 * Checks if the first GPU has sufficient compute capability for this module.
 * Issues a warning if the GPU is too old.
 */
static void checkComputeCapability() {
    // Minimum compute capability we compiled for (adjust based on your CMAKE_CUDA_ARCHITECTURES)
    constexpr int MIN_CC_MAJOR = 7;  // Based on CMAKE_CUDA_ARCHITECTURES 75
    constexpr int MIN_CC_MINOR = 5;

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return; // Skip check if no devices or can't query
    }

    // Check device 0 (primary device)
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return; // Skip if can't get properties
    }

    if (prop.major < MIN_CC_MAJOR ||
        (prop.major == MIN_CC_MAJOR && prop.minor < MIN_CC_MINOR)) {
        std::string warning_msg =
            "[mach] GPU compute capability " +
            std::to_string(prop.major) + "." + std::to_string(prop.minor) +
            " is below the minimum required " +
            std::to_string(MIN_CC_MAJOR) + "." + std::to_string(MIN_CC_MINOR) +
            ". Kernels may fail to load.";

        PyErr_WarnEx(PyExc_RuntimeWarning, warning_msg.c_str(), 1);
    }
}

/**
 * @brief Calculate the transmit+receive delay and apodization weight for a single element position
 * @param rx_coord_m: float3, the position of the receive element (m)
 * @param voxel_xyz: float3, the position of the voxel (m)
 * @param aperture_radius_squared: float, the square of the aperture radius (m^2)
 * @param aperture_radius: float, the aperture radius (m)
 * @param voxel_tx_delay_s: float, the transmit delay time (seconds, includes rx_start_s offset)
 * @param sampling_freq_hz: float, the sampling frequency (Hz)
 * @param inv_sound_speed_m_s: float, the inverse of the speed of sound in medium (seconds/meters)
 * @param tukey_alpha: float, the alpha parameter for the Tukey window
 * @param rx_start_s: float, acquisition start time, i.e. how long after transmit was the first channel_data sample
*   (corresponds to t0 in biomecardio.com/publis/ultrasonics21.pdf)
 * @tparam UseApodization: bool, whether to use apodization
 * @return float2: (physical_tau_s, apod_weight)
 *         physical_tau_s: total physical wave-travel time in seconds (for phase correction)
 *              note: slightly different definition from biomecardio.com/publis/ultrasonics21.pdf
 *              here: tau = (d_TX + d_RX) / c, (does not include t0)
 *         apod_weight: apodization weight
 */
template<bool UseApodization>
__device__ static inline float2 calculateTxRxDelayAndApodization(
    const float3 rx_coord_m,
    const float3 voxel_xyz,
    const float aperture_radius_squared,
    const float aperture_radius,
    const float voxel_tx_delay_s,
    const float sampling_freq_hz,
    const float inv_sound_speed_m_s,
    float tukey_alpha
) {
    // Compute relative position and distance
    const float dx = rx_coord_m.x - voxel_xyz.x;
    const float dy = rx_coord_m.y - voxel_xyz.y;
    const float dz = rx_coord_m.z - voxel_xyz.z;
    const float horizontal_distance_squared = dx * dx + dy * dy;

    // Skip computation if outside aperture based on F-number (early out)
    if (horizontal_distance_squared > aperture_radius_squared) {
        return make_float2(-1.0f, 0.0f);  // Return invalid marker for elements outside aperture
    }

    // Calculate physical distances and wave-travel times
    const float rx_distance = __fsqrt_rn(horizontal_distance_squared + dz * dz);
    const float rx_delay_s = rx_distance * inv_sound_speed_m_s;  // rx_distance / speed_of_sound

    // For phase calculation: physical wave-travel time in seconds (tau = tx_time + rx_time)
    const float physical_tau_s = voxel_tx_delay_s + rx_delay_s;

    if constexpr (!UseApodization) {
        return make_float2(physical_tau_s, 1.0f);
    }

    // Calculate apodization weight
    const float horizontal_distance = __fsqrt_rn(horizontal_distance_squared); // Radial distance
    // Look up the apodization weight using the Tukey window function
    const float weight = tukey_apod_weight(horizontal_distance / aperture_radius, tukey_alpha);

    return make_float2(physical_tau_s, weight);
}

/**
 * @brief CUDA kernel for delay-and-sum beamforming.
 *
 * This kernel processes ultrasound sensor data for multiple voxels and frames simultaneously.
 * Each thread processes specific frames and receive elements for a particular voxel.
 *
 * Thread Organization:
 * - Thread block dimensions: (frames, voxels) with adaptive sizing
 * - Grid dimensions: (voxel_batches_x, voxel_batches_y, receive_element_batches)
 * - Each block processes VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE voxels * receive elements
 *   By default, this is 8 voxels * 140 receive elements = 1120, although this scales
 *   to increase thread count if needed.
 *
 * Memory Access Patterns:
 * - Shared memory for delay/apodization tables (reused across frames)
 * - Coalesced memory access to sensor data
 * - Atomic operations for output accumulation across receive element batches
 *
 * @tparam DataType Either float (for RF data) or float2 (for I/Q data)
 * @tparam StorageType Storage type of channel_data: DataType (default) or its FP16 counterpart
 * @tparam UseApodization Whether to apply Tukey window apodization
 * @tparam interpType Interpolation method for sensor data sampling
 * @param channel_data Input sensor data [n_receive_elements][n_samples][frame_stride] (DataType)
 * @param n_frames Number of frames to beamform (leading frames of each sample)
 * @param frame_stride Frames allocated per sample in channel_data; >= n_frames
 * @param n_receive_elements Number of receive elements
 * @param n_samples Number of time samples per element
 * @param rx_coords_m Receive element positions [n_receive_elements] (float3 x,y,z in meters)
 * @param output_voxels_xyz Output voxel positions [n_output_voxels] (float3 x,y,z in meters)
 * @param tx_arrival_delays Transmit delays for each voxel [n_output_voxels] (in seconds)
 * @param beamformed Output beamformed data [n_output_voxels][n_frames] (DataType)
 * @param sampling_freq_hz Sampling frequency (Hz)
 * @param inv_sound_speed_m_s Inverse of speed of sound in medium (seconds/meters)
 * @param modulation_freq_hz Modulation frequency (Hz); usually the transmit center-frequency if IQ data was demodulated, 0 if RF data
 * @param f_number F-number for aperture
 * @param tukey_alpha Alpha parameter for Tukey window apodization
 * @param rx_start_s acquisition start time, i.e.  offset (seconds, corresponds to t0 in biomecardio.com/publis/ultrasonics21.pdf)
 * @param n_output_voxels Number of output voxels
 * @param receive_elements_batch_size Number of receive elements to process per batch
 */
template<typename DataType, bool UseApodization, InterpolationType interpType, typename StorageType = DataType>
__global__ void beamformKernel(
    const StorageType* const __restrict__ channel_data,
    __grid_constant__ const uint32_t n_frames,
    __grid_constant__ const uint32_t frame_stride,
    __grid_constant__ const uint32_t n_receive_elements,
    __grid_constant__ const uint32_t n_samples,
    const float3* const __restrict__ rx_coords_m,
    const float3* const __restrict__ output_voxels_xyz,
    const float* const __restrict__ tx_arrival_delays,
    DataType* __restrict__ beamformed,
    __grid_constant__ const float sampling_freq_hz,
    __grid_constant__ const float inv_sound_speed_m_s,
    __grid_constant__ const float modulation_freq_hz,
    __grid_constant__ const float f_number,
    __grid_constant__ const float tukey_alpha,
    __grid_constant__ const float rx_start_s,
    __grid_constant__ const uint64_t n_output_voxels,
    __grid_constant__ const uint32_t receive_elements_batch_size
) {
    // Ensure DataType is one of the supported types for ultrasound beamforming.
    // StorageType controls only how channel_data is stored: full precision
    // (same as DataType) or FP16 (__half for RF, __half2 for I/Q). Compute and
    // output stay FP32.
    static_assert((std::is_same_v<DataType, float> &&
                   (std::is_same_v<StorageType, float> || std::is_same_v<StorageType, __half>)) ||
                  (std::is_same_v<DataType, float2> &&
                   (std::is_same_v<StorageType, float2> || std::is_same_v<StorageType, __half2>)),
                  "DataType must be float (RF) or float2 (I/Q); StorageType must be "
                  "the same type or its FP16 counterpart (__half / __half2).");
    constexpr bool is_complex = std::is_same_v<DataType, float2>;

    // Calculate the base voxel index for this block
    // threadIdx corresponds to: (x=frame_tid, y=voxel_tid)
    // blockIdx corresponds to: (x=voxel_batch, y=extra_voxel_batch (if overflow), z=receive_element_batch)
    // Each thread processes multiple frames and receive elements
    const unsigned int frame_tid = threadIdx.x;  // Frame dimension
    const unsigned int voxel_tid = threadIdx.y;  // Voxel dimension (within block)
    const unsigned int num_frame_threads = blockDim.x;
    const unsigned int num_voxels_per_block = blockDim.y;
    const uint32_t receive_element_block_start_idx = blockIdx.z * receive_elements_batch_size;
    const unsigned int receive_elements_in_batch = min(receive_elements_batch_size, n_receive_elements - receive_element_block_start_idx);
    // CUDA: blockIdx.x <= 2**16 - 1, blockIdx.y <= 2**16 - 1, gridDim.x <= 2**16 - 1
    // so we can use uint32_t for voxel_batch_idx
    const uint32_t voxel_batch_idx = blockIdx.x + blockIdx.y * gridDim.x;
    // However, voxel_batch_idx * num_voxels_per_block may overflow uint32_t, so we use uint64_t for voxel_idx
    const uint64_t voxel_idx = static_cast<uint64_t>(voxel_batch_idx) * num_voxels_per_block + voxel_tid;  // Voxel dimension (within block)

#ifdef CUDA_DEBUG
    const uint64_t n_channel_data = static_cast<uint64_t>(n_receive_elements) * static_cast<uint64_t>(n_samples) * static_cast<uint64_t>(frame_stride);
    DEBUG_ASSERT(n_channel_data < UINT32_MAX);
#endif

    // Pre-calculate group variables for performance
    // Note: modulation_freq_hz is only used for I/Q data (float2), ignored for RF data (float)
    const float modulation_freq_rad = 2.0f * PI * modulation_freq_hz;

    // Dynamically allocated shared memory - organized as a single flat array
    // Indexing dimensions: [voxel_tid][receive_element_idx_in_batch]
    static __shared__ float2 voxel_tau_and_apod_weights[VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE];

    // Skip if we're outside the valid voxel range
    if (voxel_idx >= n_output_voxels) return;

    // Load grid point and tx delay value for this voxel
    const float3 voxel_xyz = output_voxels_xyz[voxel_idx];
    const float voxel_tx_delay_s = tx_arrival_delays[voxel_idx];
    const float aperture_radius = voxel_xyz.z / (2.0f * f_number);
    const float aperture_radius_squared = aperture_radius * aperture_radius;

    // For the delay calculation phase, we use the frame-threads to parallelize over receive elements
    // Pre-compute delays and weights, which are shared across frames, but are different for each voxel
    for (unsigned int receive_element_idx_in_batch = frame_tid; receive_element_idx_in_batch < receive_elements_in_batch; receive_element_idx_in_batch += num_frame_threads) {
        const uint32_t receive_element_idx = receive_element_block_start_idx + receive_element_idx_in_batch;
        float3 rx_coord_m = rx_coords_m[receive_element_idx];

        float2 tau_and_weight = calculateTxRxDelayAndApodization<UseApodization>(
            rx_coord_m,
            voxel_xyz,
            aperture_radius_squared,
            aperture_radius,
            voxel_tx_delay_s,
            sampling_freq_hz,
            inv_sound_speed_m_s,
            tukey_alpha
        );

        const uint32_t shared_mem_idx = voxel_tid * receive_elements_in_batch + receive_element_idx_in_batch;
        DEBUG_ASSERT(shared_mem_idx < VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE);  // Verify shared memory access is in bounds

        voxel_tau_and_apod_weights[shared_mem_idx] = tau_and_weight;

        DEBUG_ASSERT(receive_element_idx < n_receive_elements);  // Verify element index is valid
        DEBUG_ASSERT(receive_element_idx_in_batch < receive_elements_in_batch);  // Verify batch index is valid
    }
    __syncthreads();

    // Strided-loop over frames for coalesced memory access into channel_data
    for (uint32_t frame_idx = frame_tid; frame_idx < n_frames; frame_idx += num_frame_threads) {
        DataType frame_sum{};

        // Initialize frame_sum based on data type
        if constexpr (is_complex) {
            frame_sum = make_float2(0.0f, 0.0f);
        } else {
            frame_sum = 0.0f;
        }

        // Every thread (managing 1 voxel, for 1 frame per iteration)
        // Needs to sum over all receive elements in the batch
        for (uint32_t receive_element_idx_in_batch = 0; receive_element_idx_in_batch < receive_elements_in_batch; receive_element_idx_in_batch ++) {
            const uint32_t receive_element_idx = receive_element_block_start_idx + receive_element_idx_in_batch;
            const uint32_t shared_mem_idx = voxel_tid * receive_elements_in_batch + receive_element_idx_in_batch;
            const float2 tau_and_weight = voxel_tau_and_apod_weights[shared_mem_idx];
            const float physical_tau_s = tau_and_weight.x;    // Physical wave-travel time in seconds (for phase)
            const float apod_weight = tau_and_weight.y;       // Apodization weight

            // Skip if element is outside aperture (marked with physical_tau_s = -1.0f) or apodization weight is 0
            if ((physical_tau_s < 0.0f) || (apod_weight == 0.0f)) continue;

            // Receive-sample-index (float, before interpolation)
            const float sample_idx = (physical_tau_s - rx_start_s) * sampling_freq_hz;

            // Use template-based interpolation dispatch with unified bounds checking
            bool is_valid;
            DataType sensor_sample = interpolate_sample<DataType, interpType, StorageType>(
                channel_data, sample_idx, receive_element_idx, frame_idx, n_samples, frame_stride, is_valid
            );

            // Skip if sample is outside bounds
            if (!is_valid) continue;

            if constexpr (UseApodization) {
                sensor_sample *= apod_weight;
            }

            // Process and accumulate the sample data
            if constexpr (is_complex) {
                if (modulation_freq_hz != 0.0f) {
                    // Phase-shift I/Q data using physical wave-travel time (tau)
                    const float phi = modulation_freq_rad * physical_tau_s;
                    float cos_phi, sin_phi;
                    __sincosf(phi, &sin_phi, &cos_phi);
                    float shifted_sample_real = fmaf(sensor_sample.x, cos_phi, fmaf(sensor_sample.y, -sin_phi, 0.0f));
                    float shifted_sample_imag = fmaf(sensor_sample.y, cos_phi, fmaf(sensor_sample.x, sin_phi, 0.0f));
                    frame_sum += make_float2(shifted_sample_real, shifted_sample_imag);
                } else {
                    // Special case for modulation_freq_hz = 0
                    frame_sum += sensor_sample;
                }
            } else {
                // Real data - just accumulate (modulation_freq_hz is ignored for RF data)
                frame_sum += sensor_sample;
            }
        }
        const uint64_t beamformed_idx = static_cast<uint64_t>(voxel_idx) * static_cast<uint64_t>(n_frames) + static_cast<uint64_t>(frame_idx);
        DEBUG_ASSERT(beamformed_idx < static_cast<uint64_t>(n_output_voxels) * static_cast<uint64_t>(n_frames));  // Verify beamformed output index is valid

        // Use appropriate atomic add based on data type
        if constexpr (is_complex) {
            atomicAdd(&beamformed[beamformed_idx].x, frame_sum.x);
            atomicAdd(&beamformed[beamformed_idx].y, frame_sum.y);
        } else {
            atomicAdd(&beamformed[beamformed_idx], frame_sum);
        }
    }
}

/**
 * @brief Inverted-loop I/Q beamforming kernel (frames innermost).
 *
 * Restructures beamformKernel's phase 2: instead of walking the receive-element
 * loop once per frame, each thread owns FPT consecutive frames and the element
 * loop runs once, hoisting per-element work (sample index, bounds check,
 * apodization, phase-rotation sincos) out of the frame dimension. The FPT
 * frames are fetched with 128-bit vector loads into independent accumulators,
 * converting the serial load->FMA dependency chain into ILP-rich streaming.
 * Phase 1 (shared delay/apod table) is identical to beamformKernel.
 *
 * Summation order differs from beamformKernel (apodization and phase rotation
 * are folded into one complex weight per element), so results agree to FP32
 * rounding (~1e-6 relative), not bitwise.
 *
 * @tparam StorageType float2 (FP32 storage) or __half2 (FP16 storage)
 * @tparam UseApodization Whether to apply Tukey apodization
 * @tparam interpType NearestNeighbor or Linear (Quadratic uses beamformKernel)
 * @tparam FPT Frames per thread (4 == one 128-bit load per tap for __half2)
 */
template<typename StorageType, bool UseApodization, InterpolationType interpType, int FPT>
__global__ void beamformKernelInvIQ(
    const StorageType* const __restrict__ channel_data,
    __grid_constant__ const uint32_t n_frames,
    __grid_constant__ const uint32_t frame_stride,
    __grid_constant__ const uint32_t n_receive_elements,
    __grid_constant__ const uint32_t n_samples,
    const float3* const __restrict__ rx_coords_m,
    const float3* const __restrict__ output_voxels_xyz,
    const float* const __restrict__ tx_arrival_delays,
    float2* __restrict__ beamformed,
    __grid_constant__ const float sampling_freq_hz,
    __grid_constant__ const float inv_sound_speed_m_s,
    __grid_constant__ const float modulation_freq_hz,
    __grid_constant__ const float f_number,
    __grid_constant__ const float tukey_alpha,
    __grid_constant__ const float rx_start_s,
    __grid_constant__ const uint64_t n_output_voxels,
    __grid_constant__ const uint32_t receive_elements_batch_size
) {
    static_assert(std::is_same_v<StorageType, float2> || std::is_same_v<StorageType, __half2>,
                  "Inverted kernel is I/Q only (float2 or __half2 storage).");
    static_assert(interpType != InterpolationType::Quadratic,
                  "Inverted kernel supports nearest/linear only; quadratic falls back.");

    const unsigned int frame_tid = threadIdx.x;
    const unsigned int voxel_tid = threadIdx.y;
    const unsigned int num_frame_threads = blockDim.x;
    const unsigned int num_voxels_per_block = blockDim.y;
    const uint32_t receive_element_block_start_idx = blockIdx.z * receive_elements_batch_size;
    const unsigned int receive_elements_in_batch = min(receive_elements_batch_size, n_receive_elements - receive_element_block_start_idx);
    const uint32_t voxel_batch_idx = blockIdx.x + blockIdx.y * gridDim.x;
    const uint64_t voxel_idx = static_cast<uint64_t>(voxel_batch_idx) * num_voxels_per_block + voxel_tid;

    const float modulation_freq_rad = 2.0f * PI * modulation_freq_hz;

    static __shared__ float2 voxel_tau_and_apod_weights[VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE];

    if (voxel_idx >= n_output_voxels) return;

    const float3 voxel_xyz = output_voxels_xyz[voxel_idx];
    const float voxel_tx_delay_s = tx_arrival_delays[voxel_idx];
    const float aperture_radius = voxel_xyz.z / (2.0f * f_number);
    const float aperture_radius_squared = aperture_radius * aperture_radius;

    // Phase 1: identical to beamformKernel
    for (unsigned int e = frame_tid; e < receive_elements_in_batch; e += num_frame_threads) {
        const uint32_t receive_element_idx = receive_element_block_start_idx + e;
        const float3 rx_coord_m = rx_coords_m[receive_element_idx];
        voxel_tau_and_apod_weights[voxel_tid * receive_elements_in_batch + e] =
            calculateTxRxDelayAndApodization<UseApodization>(
                rx_coord_m, voxel_xyz, aperture_radius_squared, aperture_radius,
                voxel_tx_delay_s, sampling_freq_hz, inv_sound_speed_m_s, tukey_alpha);
    }
    __syncthreads();

    // Phase 2: element loop outermost, FPT frames per thread innermost.
    // A trailing partial chunk still vector-loads a whole FPT frames -- the
    // dispatcher guarantees frame_stride has room for them -- and drops the
    // frames past n_frames at the store, so the element loop stays branch-free.
    const uint32_t n_chunks = (n_frames + FPT - 1) / FPT;
    for (uint32_t chunk = frame_tid; chunk < n_chunks; chunk += num_frame_threads) {
        const uint32_t frame0 = chunk * FPT;
        float2 acc[FPT];
        #pragma unroll
        for (int k = 0; k < FPT; k++) acc[k] = make_float2(0.0f, 0.0f);

        for (uint32_t e = 0; e < receive_elements_in_batch; e++) {
            const float2 tau_and_weight = voxel_tau_and_apod_weights[voxel_tid * receive_elements_in_batch + e];
            const float physical_tau_s = tau_and_weight.x;
            const float apod_weight = tau_and_weight.y;
            if ((physical_tau_s < 0.0f) || (apod_weight == 0.0f)) continue;

            const float sample_idx = (physical_tau_s - rx_start_s) * sampling_freq_hz;
            const uint32_t receive_element_idx = receive_element_block_start_idx + e;

            // Sample bounds and loads: same conditions and rounding intrinsics as
            // interpolate_nearest / interpolate_linear, one vector load per tap.
            float2 samp[FPT];
            if constexpr (interpType == InterpolationType::NearestNeighbor) {
                if ((sample_idx < -0.5f) || (sample_idx > (n_samples - 0.5f))) continue;
                const unsigned int s0 = __float2uint_rn(sample_idx);
                load_frames<StorageType, FPT>(
                    channel_data + (receive_element_idx * n_samples + s0) * frame_stride + frame0, samp);
            } else {  // Linear
                if ((sample_idx < 0.0f) || (sample_idx > (n_samples - 1))) continue;
                const unsigned int sample_idx_floor = __float2uint_rd(sample_idx);
                const unsigned int sample_idx_ceil = __float2uint_ru(sample_idx);
                const float lerp_alpha = sample_idx - (float)sample_idx_floor;
                const uint32_t base = receive_element_idx * n_samples * frame_stride + frame0;
                float2 lo[FPT], hi[FPT];
                load_frames<StorageType, FPT>(channel_data + base + sample_idx_floor * frame_stride, lo);
                load_frames<StorageType, FPT>(channel_data + base + sample_idx_ceil * frame_stride, hi);
                #pragma unroll
                for (int k = 0; k < FPT; k++) samp[k] = lerp(lo[k], hi[k], lerp_alpha);
            }

            // Apodization and phase rotation once per element, folded into one
            // complex weight (w*cos, w*sin): rot(w*s) == w*rot(s).
            const float w = UseApodization ? apod_weight : 1.0f;
            if (modulation_freq_hz != 0.0f) {
                float cos_phi, sin_phi;
                __sincosf(modulation_freq_rad * physical_tau_s, &sin_phi, &cos_phi);
                const float cw = cos_phi * w;
                const float sw = sin_phi * w;
                #pragma unroll
                for (int k = 0; k < FPT; k++) {
                    acc[k].x = fmaf(samp[k].x, cw, fmaf(-samp[k].y, sw, acc[k].x));
                    acc[k].y = fmaf(samp[k].y, cw, fmaf(samp[k].x, sw, acc[k].y));
                }
            } else {
                #pragma unroll
                for (int k = 0; k < FPT; k++) {
                    acc[k].x = fmaf(samp[k].x, w, acc[k].x);
                    acc[k].y = fmaf(samp[k].y, w, acc[k].y);
                }
            }
        }

        // beamformed is accumulated across receive-element batches, so each frame
        // must be added exactly once: drop the padding lanes rather than clamping
        // them onto a real frame.
        const uint64_t out_base = voxel_idx * static_cast<uint64_t>(n_frames) + frame0;
        #pragma unroll
        for (int k = 0; k < FPT; k++) {
            if (frame0 + k >= n_frames) break;
            atomicAdd(&beamformed[out_base + k].x, acc[k].x);
            atomicAdd(&beamformed[out_base + k].y, acc[k].y);
        }
    }
}

/**
 * @brief Launch the inverted-loop kernel if the configuration allows it.
 *
 * The 128-bit loads constrain frame_stride, not n_frames: the stride has to keep
 * every sample row 16-byte aligned and has to leave a whole FPT-frame chunk for
 * the last, possibly partial, chunk to load. Callers that allocate channel_data
 * with a padded stride therefore keep the fast path at any n_frames.
 *
 * @return false (nothing launched) when the configuration requires the original
 * kernel: RF data, quadratic interpolation, a frame_stride that is misaligned or
 * has no room for the trailing chunk, a channel_data pointer that is not
 * 16-byte aligned (an offset view), or use_inverted_kernel == false (A/B
 * comparisons against the original).
 */
template<typename DataType, typename StorageType>
bool _try_beamform_inverted(
    const StorageType* d_channel_data,
    const float3* d_rx_coords_m,
    const float3* d_scan_coords_m,
    const float* d_tx_arrivals_s,
    DataType* d_out,
    uint32_t n_receive_elements,
    uint32_t n_samples,
    uint64_t n_output_voxels,
    uint32_t n_frames,
    uint32_t frame_stride,
    float f_number,
    float rx_start_s,
    float sampling_freq_hz,
    float inv_sound_speed_m_s,
    float modulation_freq_hz,
    float tukey_alpha,
    InterpolationType interp_type,
    bool use_inverted_kernel
) {
    if constexpr (!std::is_same_v<DataType, float2>) {
        return false;
    } else {
        constexpr int FPT = 4;
        if (interp_type == InterpolationType::Quadratic) return false;
        if (n_frames == 0) return false;
        if (!use_inverted_kernel) return false;
        if (reinterpret_cast<std::uintptr_t>(d_channel_data) % 16 != 0) return false;  // 128-bit loads need a 16B base
        // Every sample row starts at a multiple of frame_stride, so an unaligned
        // stride misaligns all but the first row however the base is aligned.
        if ((frame_stride * sizeof(StorageType)) % 16 != 0) return false;
        // The trailing chunk vector-loads FPT frames whether or not n_frames fills them.
        if (((n_frames + FPT - 1) / FPT) * FPT > frame_stride) return false;

        const uint32_t n_chunks = (n_frames + FPT - 1) / FPT;
        const int frame_threads = min(static_cast<int>(n_chunks), MAX_FRAME_THREADS_PER_BLOCK);
        const int voxels_per_block = calculate_voxels_per_block(frame_threads);
        dim3 threads_per_block(frame_threads, voxels_per_block);
        DEBUG_ASSERT(threads_per_block.x * threads_per_block.y <= 1024);
        const int receive_elements_batch_size = calculate_receive_elements_batch_size(voxels_per_block);

        const int num_blocks = (n_output_voxels + voxels_per_block - 1) / voxels_per_block;
        const int max_blocks_per_dim = (1 << 16) - 32;
        const int grid_x = min(max_blocks_per_dim, num_blocks);
        const int grid_y = (num_blocks + grid_x - 1) / grid_x;
        const int grid_z = (n_receive_elements + receive_elements_batch_size - 1) / receive_elements_batch_size;
        dim3 grid(grid_x, grid_y, grid_z);

        const bool apod_flag = tukey_alpha > 0.0f;
        const bool nearest = interp_type == InterpolationType::NearestNeighbor;

        auto launch = [&](auto kernel) {
            checkCudaErrors(cudaFuncSetCacheConfig(kernel, CACHE_CONFIG));
            kernel<<<grid, threads_per_block>>>(
                d_channel_data, n_frames, frame_stride, n_receive_elements, n_samples,
                d_rx_coords_m, d_scan_coords_m, d_tx_arrivals_s, d_out,
                sampling_freq_hz, inv_sound_speed_m_s, modulation_freq_hz,
                f_number, tukey_alpha, rx_start_s, n_output_voxels,
                receive_elements_batch_size);
            checkCudaErrors(cudaGetLastError());
        };
        if (apod_flag && nearest)       launch(beamformKernelInvIQ<StorageType, true,  InterpolationType::NearestNeighbor, FPT>);
        else if (apod_flag)             launch(beamformKernelInvIQ<StorageType, true,  InterpolationType::Linear,          FPT>);
        else if (nearest)               launch(beamformKernelInvIQ<StorageType, false, InterpolationType::NearestNeighbor, FPT>);
        else                            launch(beamformKernelInvIQ<StorageType, false, InterpolationType::Linear,          FPT>);

        // Wait for kernel to complete
        checkCudaErrors(cudaDeviceSynchronize());
        return true;
    }
}

// ---------------------------------------------------------------------------
// Backward (vector-Jacobian product) of the inverted-loop I/Q kernel
// ---------------------------------------------------------------------------

/**
 * @brief Vector-Jacobian product of beamformKernelInvIQ (frames innermost).
 *
 * Forward: out[v,f] = sum_e W(v,e) * s(v,e)[f], with W = w_apod * exp(j*omega*tau),
 * s the interpolated sample at idx = (tau - rx_start_s) * fs and
 * tau = tx_arrival[v] + |rx[e] - scan[v]| / c.
 *
 * Given grad_out[v,f] = dL/dRe(out) + j*dL/dIm(out) (PyTorch's convention for a real
 * loss L), every enabled output is accumulated (+=) into a caller-zeroed buffer:
 *
 *   NeedGradData  grad_channel_data[e, n, f] += conj(W) * c_n * grad_out[v,f]
 *                 (c_n = interpolation coefficient of tap n: the adjoint of the linear map
 *                 channel_data -> out, i.e. backprojection)
 *   NeedGradTau   G_phase(v,e) = sum_f Re(conj(g) * j*W*s)        [phase-rotation term]
 *                 G_idx(v,e)   = sum_f Re(conj(g) * W*(hi - lo))  [interpolant slope, linear only]
 *                 dL/dtau(v,e) = omega * G_phase + fs * G_idx, then the chain rule:
 *                   grad_tx_arrivals[v] += dL/dtau
 *                   grad_scan_coords[v] += dL/dtau * -(rx - scan) / (r * c)
 *                   grad_rx_coords[e]   += dL/dtau *  (rx - scan) / (r * c)
 *                   grad_sound_speed    += dL/dtau * -r / c^2
 *                   grad_rx_start_s     += -fs * G_idx
 *                 The aperture mask, the sample-bounds masks and the apodization weight are
 *                 treated as constants with respect to the geometry.
 *
 * Loop order: element outermost, the thread's frame chunks innermost. Per element the
 * delay, taps and complex weight are computed once; per chunk the FPT frames of grad_out
 * (and of the two taps when NeedGradTau) are loaded and either scattered with atomics
 * (grad_channel_data) or dotted into per-element registers. Phase 1 (shared delay/apod
 * table) is identical to beamformKernelInvIQ. No thread returns early, so that the
 * block-wide phases see every thread.
 *
 * @tparam StorageType float2 (FP32 storage) or __half2 (FP16 storage) for channel_data
 * @tparam NeedGradData accumulate grad_channel_data
 * @tparam NeedGradTau  accumulate the geometry / sound-speed / rx_start_s gradients
 */
template<typename StorageType, bool UseApodization, InterpolationType interpType, int FPT,
         bool NeedGradData, bool NeedGradTau>
__global__ void beamformKernelInvIQVJP(
    const StorageType* const __restrict__ channel_data,
    const float2* const __restrict__ grad_out,
    float2* __restrict__ grad_channel_data,
    float* __restrict__ grad_tx_arrivals,
    float3* __restrict__ grad_scan_coords,
    float3* __restrict__ grad_rx_coords,
    double* __restrict__ grad_sound_speed,
    double* __restrict__ grad_rx_start_s,
    __grid_constant__ const uint32_t n_frames,
    __grid_constant__ const uint32_t frame_stride,
    __grid_constant__ const uint32_t n_receive_elements,
    __grid_constant__ const uint32_t n_samples,
    const float3* const __restrict__ rx_coords_m,
    const float3* const __restrict__ output_voxels_xyz,
    const float* const __restrict__ tx_arrival_delays,
    __grid_constant__ const float sampling_freq_hz,
    __grid_constant__ const float inv_sound_speed_m_s,
    __grid_constant__ const float modulation_freq_hz,
    __grid_constant__ const float f_number,
    __grid_constant__ const float tukey_alpha,
    __grid_constant__ const float rx_start_s,
    __grid_constant__ const uint64_t n_output_voxels,
    __grid_constant__ const uint32_t receive_elements_batch_size
) {
    static_assert(NeedGradData || NeedGradTau, "VJP kernel instantiated with nothing to compute.");
    static_assert(std::is_same_v<StorageType, float2> || std::is_same_v<StorageType, __half2>,
                  "VJP kernel is I/Q only (float2 or __half2 storage).");
    static_assert(interpType != InterpolationType::Quadratic,
                  "VJP kernel supports nearest/linear only.");

    const unsigned int frame_tid = threadIdx.x;
    const unsigned int voxel_tid = threadIdx.y;
    const unsigned int num_frame_threads = blockDim.x;
    const unsigned int num_voxels_per_block = blockDim.y;
    const unsigned int tid = voxel_tid * num_frame_threads + frame_tid;
    const unsigned int n_threads = num_frame_threads * num_voxels_per_block;
    const uint32_t receive_element_block_start_idx = blockIdx.z * receive_elements_batch_size;
    const unsigned int receive_elements_in_batch = min(receive_elements_batch_size, n_receive_elements - receive_element_block_start_idx);
    const uint32_t voxel_batch_idx = blockIdx.x + blockIdx.y * gridDim.x;
    const uint64_t voxel_idx = static_cast<uint64_t>(voxel_batch_idx) * num_voxels_per_block + voxel_tid;
    const bool voxel_valid = voxel_idx < n_output_voxels;
    const float modulation_freq_rad = 2.0f * PI * modulation_freq_hz;

    static __shared__ float2 voxel_tau_and_apod_weights[VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE];
    __shared__ float g_phase_table[NeedGradTau ? VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE : 1];
    __shared__ float g_idx_table[NeedGradTau ? VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE : 1];
    __shared__ double block_grad_sound_speed;
    __shared__ double block_grad_rx_start_s;

    // Phase 1: delay/apod table (identical to the forward), plus zeroed reduction tables.
    if (voxel_valid) {
        const float3 voxel_xyz = output_voxels_xyz[voxel_idx];
        const float voxel_tx_delay_s = tx_arrival_delays[voxel_idx];
        const float aperture_radius = voxel_xyz.z / (2.0f * f_number);
        const float aperture_radius_squared = aperture_radius * aperture_radius;
        for (unsigned int e = frame_tid; e < receive_elements_in_batch; e += num_frame_threads) {
            const uint32_t receive_element_idx = receive_element_block_start_idx + e;
            voxel_tau_and_apod_weights[voxel_tid * receive_elements_in_batch + e] =
                calculateTxRxDelayAndApodization<UseApodization>(
                    rx_coords_m[receive_element_idx], voxel_xyz, aperture_radius_squared, aperture_radius,
                    voxel_tx_delay_s, sampling_freq_hz, inv_sound_speed_m_s, tukey_alpha);
        }
    }
    if constexpr (NeedGradTau) {
        for (unsigned int i = tid; i < num_voxels_per_block * receive_elements_in_batch; i += n_threads) {
            g_phase_table[i] = 0.0f;
            g_idx_table[i] = 0.0f;
        }
        if (tid == 0) {
            block_grad_sound_speed = 0.0;
            block_grad_rx_start_s = 0.0;
        }
    }
    __syncthreads();

    // Phase 2: element loop outermost, this thread's frame chunks innermost.
    const uint32_t n_chunks = (n_frames + FPT - 1) / FPT;
    if (voxel_valid) {
        const float2* const g_row = grad_out + voxel_idx * static_cast<uint64_t>(n_frames);
        for (uint32_t e = 0; e < receive_elements_in_batch; e++) {
            const float2 tau_and_weight = voxel_tau_and_apod_weights[voxel_tid * receive_elements_in_batch + e];
            const float physical_tau_s = tau_and_weight.x;
            const float apod_weight = tau_and_weight.y;
            if ((physical_tau_s < 0.0f) || (apod_weight == 0.0f)) continue;
            const float sample_idx = (physical_tau_s - rx_start_s) * sampling_freq_hz;
            const uint32_t receive_element_idx = receive_element_block_start_idx + e;

            // Taps (same bounds tests and rounding intrinsics as the forward): rows row0/row1
            // with coefficients (1 - coef1)/coef1; nearest neighbour uses row0 only.
            uint32_t row0, row1;
            float coef1;
            if constexpr (interpType == InterpolationType::NearestNeighbor) {
                if ((sample_idx < -0.5f) || (sample_idx > (n_samples - 0.5f))) continue;
                row0 = __float2uint_rn(sample_idx);
                row1 = row0;
                coef1 = 0.0f;
            } else {
                if ((sample_idx < 0.0f) || (sample_idx > (n_samples - 1))) continue;
                row0 = __float2uint_rd(sample_idx);
                row1 = __float2uint_ru(sample_idx);
                coef1 = sample_idx - (float)row0;
            }
            const float coef0 = 1.0f - coef1;

            // Complex weight W = w * exp(j*phi) as (cw, sw), once per element.
            const float w = UseApodization ? apod_weight : 1.0f;
            float cw = w, sw = 0.0f;
            if (modulation_freq_hz != 0.0f) {
                float cos_phi, sin_phi;
                __sincosf(modulation_freq_rad * physical_tau_s, &sin_phi, &cos_phi);
                cw = cos_phi * w;
                sw = sin_phi * w;
            }
            const size_t elem_base = static_cast<size_t>(receive_element_idx) * n_samples * frame_stride;
            const size_t row0_base = elem_base + static_cast<size_t>(row0) * frame_stride;
            const size_t row1_base = elem_base + static_cast<size_t>(row1) * frame_stride;

            float g_phase = 0.0f, g_idx = 0.0f;
            for (uint32_t chunk = frame_tid; chunk < n_chunks; chunk += num_frame_threads) {
                const uint32_t frame0 = chunk * FPT;
                // grad_out rows are n_frames long (not padded): scalar loads, masked past n_frames.
                float2 g[FPT];
                #pragma unroll
                for (int k = 0; k < FPT; k++) {
                    g[k] = (frame0 + k < n_frames) ? g_row[frame0 + k] : make_float2(0.0f, 0.0f);
                }
                if constexpr (NeedGradData) {
                    float2* const grad_row0 = grad_channel_data + row0_base + frame0;
                    float2* const grad_row1 = grad_channel_data + row1_base + frame0;
                    #pragma unroll
                    for (int k = 0; k < FPT; k++) {
                        if (frame0 + k >= n_frames) break;
                        // conj(W) * g
                        const float vx = fmaf(cw, g[k].x, sw * g[k].y);
                        const float vy = fmaf(cw, g[k].y, -sw * g[k].x);
                        atomicAdd(&grad_row0[k].x, coef0 * vx);
                        atomicAdd(&grad_row0[k].y, coef0 * vy);
                        if constexpr (interpType == InterpolationType::Linear) {
                            atomicAdd(&grad_row1[k].x, coef1 * vx);
                            atomicAdd(&grad_row1[k].y, coef1 * vy);
                        }
                    }
                }
                if constexpr (NeedGradTau) {
                    float2 lo[FPT];
                    load_frames<StorageType, FPT>(channel_data + row0_base + frame0, lo);
                    float2 hi[FPT];
                    if constexpr (interpType == InterpolationType::Linear) {
                        load_frames<StorageType, FPT>(channel_data + row1_base + frame0, hi);
                    }
                    #pragma unroll
                    for (int k = 0; k < FPT; k++) {
                        float2 s = lo[k];
                        float2 d = make_float2(0.0f, 0.0f);
                        if constexpr (interpType == InterpolationType::Linear) {
                            d = hi[k] - lo[k];
                            s = lo[k] + coef1 * d;
                        }
                        // h = W*s; Re(conj(g) * j*h) = -Im(conj(g)*h) = g.y*h.x - g.x*h.y
                        const float hx = cw * s.x - sw * s.y;
                        const float hy = cw * s.y + sw * s.x;
                        g_phase = fmaf(g[k].y, hx, fmaf(-g[k].x, hy, g_phase));
                        if constexpr (interpType == InterpolationType::Linear) {
                            // Re(conj(g) * W*d) = g.x*hd.x + g.y*hd.y
                            const float hdx = cw * d.x - sw * d.y;
                            const float hdy = cw * d.y + sw * d.x;
                            g_idx = fmaf(g[k].x, hdx, fmaf(g[k].y, hdy, g_idx));
                        }
                    }
                }
            }
            if constexpr (NeedGradTau) {
                atomicAdd(&g_phase_table[voxel_tid * receive_elements_in_batch + e], g_phase);
                if constexpr (interpType == InterpolationType::Linear) {
                    atomicAdd(&g_idx_table[voxel_tid * receive_elements_in_batch + e], g_idx);
                }
            }
        }
    }

    if constexpr (NeedGradTau) {
        __syncthreads();
        // Phase 3: chain rule from dL/dtau(v,e) to the geometry, cooperatively over the block's table.
        const float inv_c2 = inv_sound_speed_m_s * inv_sound_speed_m_s;
        double local_grad_c = 0.0, local_grad_t0 = 0.0;
        const unsigned int n_entries = num_voxels_per_block * receive_elements_in_batch;
        for (unsigned int i = tid; i < n_entries; i += n_threads) {
            const unsigned int v_local = i / receive_elements_in_batch;
            const unsigned int e = i - v_local * receive_elements_in_batch;
            const uint64_t v = static_cast<uint64_t>(voxel_batch_idx) * num_voxels_per_block + v_local;
            if (v >= n_output_voxels) continue;
            const float2 tau_and_weight = voxel_tau_and_apod_weights[i];
            if ((tau_and_weight.x < 0.0f) || (tau_and_weight.y == 0.0f)) continue;  // outside the aperture
            const float gp = g_phase_table[i];
            const float gi = g_idx_table[i];
            const float g_tau = fmaf(modulation_freq_rad, gp, sampling_freq_hz * gi);
            if ((g_tau == 0.0f) && (gi == 0.0f)) continue;
            const uint32_t receive_element_idx = receive_element_block_start_idx + e;
            const float3 rx = rx_coords_m[receive_element_idx];
            const float3 vx = output_voxels_xyz[v];
            const float dx = rx.x - vx.x, dy = rx.y - vx.y, dz = rx.z - vx.z;
            const float r = sqrtf(dx * dx + dy * dy + dz * dz);
            // dL/dtau * dtau/dr / r, with dtau/dr = 1/c: the coordinate gradients are +-k * (dx, dy, dz)
            const float k = (r > 0.0f) ? g_tau * inv_sound_speed_m_s / r : 0.0f;
            if (grad_tx_arrivals != nullptr) atomicAdd(&grad_tx_arrivals[v], g_tau);
            if (grad_scan_coords != nullptr) {
                atomicAdd(&grad_scan_coords[v].x, -k * dx);
                atomicAdd(&grad_scan_coords[v].y, -k * dy);
                atomicAdd(&grad_scan_coords[v].z, -k * dz);
            }
            if (grad_rx_coords != nullptr) {
                atomicAdd(&grad_rx_coords[receive_element_idx].x, k * dx);
                atomicAdd(&grad_rx_coords[receive_element_idx].y, k * dy);
                atomicAdd(&grad_rx_coords[receive_element_idx].z, k * dz);
            }
            local_grad_c += static_cast<double>(g_tau) * static_cast<double>(-r * inv_c2);
            local_grad_t0 += static_cast<double>(gi) * static_cast<double>(-sampling_freq_hz);
        }
        if (grad_sound_speed != nullptr) atomicAdd(&block_grad_sound_speed, local_grad_c);
        if (grad_rx_start_s != nullptr) atomicAdd(&block_grad_rx_start_s, local_grad_t0);
        __syncthreads();
        if (tid == 0) {
            if (grad_sound_speed != nullptr) atomicAdd(grad_sound_speed, block_grad_sound_speed);
            if (grad_rx_start_s != nullptr) atomicAdd(grad_rx_start_s, block_grad_rx_start_s);
        }
    }
}

/**
 * @brief Launch the VJP kernel. Requires the inverted-kernel layout (see
 * _try_beamform_inverted); throws std::runtime_error otherwise, since there is no
 * per-frame fallback for the backward pass. Null gradient pointers are skipped.
 */
template<typename StorageType>
void _beamform_inverted_vjp(
    const StorageType* d_channel_data,
    const float2* d_grad_out,
    float2* d_grad_channel_data,
    float* d_grad_tx_arrivals,
    float3* d_grad_scan_coords,
    float3* d_grad_rx_coords,
    double* d_grad_sound_speed,
    double* d_grad_rx_start_s,
    const float3* d_rx_coords_m,
    const float3* d_scan_coords_m,
    const float* d_tx_arrivals_s,
    uint32_t n_receive_elements,
    uint32_t n_samples,
    uint64_t n_output_voxels,
    uint32_t n_frames,
    uint32_t frame_stride,
    float f_number,
    float rx_start_s,
    float sampling_freq_hz,
    float sound_speed_m_s,
    float modulation_freq_hz,
    float tukey_alpha,
    InterpolationType interp_type
) {
    constexpr int FPT = 4;
    const bool need_data = d_grad_channel_data != nullptr;
    const bool need_tau = (d_grad_tx_arrivals != nullptr) || (d_grad_scan_coords != nullptr) ||
                          (d_grad_rx_coords != nullptr) || (d_grad_sound_speed != nullptr) ||
                          (d_grad_rx_start_s != nullptr);
    if (!need_data && !need_tau) return;
    if (interp_type == InterpolationType::Quadratic) {
        throw std::runtime_error("beamform_vjp: quadratic interpolation has no backward kernel (use nearest or linear)");
    }
    if (n_frames == 0) return;
    if (reinterpret_cast<std::uintptr_t>(d_channel_data) % 16 != 0) {
        throw std::runtime_error("beamform_vjp: channel_data must be 16-byte aligned (not an offset view)");
    }
    if ((frame_stride * sizeof(StorageType)) % 16 != 0) {
        throw std::runtime_error("beamform_vjp: channel_data.shape[2] (the frame stride) must keep every sample row "
                                 "16-byte aligned: a multiple of " + std::to_string(16 / sizeof(StorageType)) + " frames");
    }
    if (((n_frames + FPT - 1) / FPT) * FPT > frame_stride) {
        throw std::runtime_error("beamform_vjp: channel_data.shape[2] must leave room for a whole " + std::to_string(FPT) +
                                 "-frame chunk past grad_out.shape[1] (pad the frame stride up to a multiple of " +
                                 std::to_string(FPT) + ")");
    }
    const uint32_t n_chunks = (n_frames + FPT - 1) / FPT;
    const int frame_threads = min(static_cast<int>(n_chunks), MAX_FRAME_THREADS_PER_BLOCK);
    const int voxels_per_block = calculate_voxels_per_block(frame_threads);
    dim3 threads_per_block(frame_threads, voxels_per_block);
    const int receive_elements_batch_size = calculate_receive_elements_batch_size(voxels_per_block);
    const int num_blocks = (n_output_voxels + voxels_per_block - 1) / voxels_per_block;
    const int max_blocks_per_dim = (1 << 16) - 32;
    const int grid_x = min(max_blocks_per_dim, num_blocks);
    const int grid_y = (num_blocks + grid_x - 1) / grid_x;
    const int grid_z = (n_receive_elements + receive_elements_batch_size - 1) / receive_elements_batch_size;
    dim3 grid(grid_x, grid_y, grid_z);
    const float inv_sound_speed_m_s = 1.0f / sound_speed_m_s;
    const bool apod_flag = tukey_alpha > 0.0f;
    const bool nearest = interp_type == InterpolationType::NearestNeighbor;
    auto launch = [&](auto kernel) {
        checkCudaErrors(cudaFuncSetCacheConfig(kernel, CACHE_CONFIG));
        kernel<<<grid, threads_per_block>>>(
            d_channel_data, d_grad_out, d_grad_channel_data, d_grad_tx_arrivals, d_grad_scan_coords,
            d_grad_rx_coords, d_grad_sound_speed, d_grad_rx_start_s,
            n_frames, frame_stride, n_receive_elements, n_samples,
            d_rx_coords_m, d_scan_coords_m, d_tx_arrivals_s,
            sampling_freq_hz, inv_sound_speed_m_s, modulation_freq_hz,
            f_number, tukey_alpha, rx_start_s, n_output_voxels,
            receive_elements_batch_size);
        checkCudaErrors(cudaGetLastError());
    };
#define MACH_VJP_LAUNCH(APOD, INTERP)                                                                   \
    if (need_data && need_tau) launch(beamformKernelInvIQVJP<StorageType, APOD, INTERP, FPT, true, true>);   \
    else if (need_data)        launch(beamformKernelInvIQVJP<StorageType, APOD, INTERP, FPT, true, false>);  \
    else                       launch(beamformKernelInvIQVJP<StorageType, APOD, INTERP, FPT, false, true>);
    if (apod_flag && nearest)  { MACH_VJP_LAUNCH(true,  InterpolationType::NearestNeighbor) }
    else if (apod_flag)        { MACH_VJP_LAUNCH(true,  InterpolationType::Linear) }
    else if (nearest)          { MACH_VJP_LAUNCH(false, InterpolationType::NearestNeighbor) }
    else                       { MACH_VJP_LAUNCH(false, InterpolationType::Linear) }
#undef MACH_VJP_LAUNCH
    checkCudaErrors(cudaDeviceSynchronize());
}

/**
 * @brief Beamforming function template wrapper that calls the appropriate kernel based on the data type.
 *
 * This function sets up the CUDA environment and calls the beamformKernel to perform delay-and-sum beamforming.
 * It handles both float (RF data) and float2 (I/Q data) variants.

 * The implementation uses CUDA with:
 * - One block processes multiple output voxels
 * - Thread dimensions: (frames, voxels)
 * - Shared memory for delay and apodization tables
 * - Coalesced memory access patterns
 *
 * @tparam DataType Either float (for RF data) or float2 (for I/Q data)
 * @tparam StorageType Storage type of channel_data: DataType (default) or its FP16 counterpart
 * @param d_channel_data Device pointer to sensor data [n_receive_elements, n_samples, frame_stride]
 * @param d_rx_coords_m Device pointer to receive element positions [n_receive_elements, 3]
 * @param d_scan_coords_m Device pointer to output voxel positions [n_output_voxels, 3]
 * @param d_tx_arrivals_s Device pointer to transmit delays [n_output_voxels]
 * @param d_out Device pointer to output beamformed data [n_output_voxels, n_frames]
 * @param n_receive_elements Number of receive elements
 * @param n_samples Number of time samples
 * @param n_output_voxels Number of output voxels
 * @param n_frames Number of frames to beamform
 * @param frame_stride Frames allocated per sample in d_channel_data; >= n_frames
 * @param f_number F-number for aperture growth control
 * @param rx_start_s Receive start time offset (seconds, corresponds to t0 in biomecardio.com/publis/ultrasonics21.pdf)
 * @param sampling_freq_hz Sampling frequency of channel_data (Hz)
 * @param sound_speed_m_s Speed of sound in medium (meters/second)
 * @param modulation_freq_hz Modulation frequency (Hz)
 * @param tukey_alpha Tukey window alpha for apodization (0=no apodization, 1=full apodization)
 * @param interp_type Interpolation method for sensor data sampling
 */
template<typename DataType, typename StorageType = DataType>
void _beamform_impl(
    const StorageType* d_channel_data,
    const float3* d_rx_coords_m,
    const float3* d_scan_coords_m,
    const float* d_tx_arrivals_s,
    DataType* d_out,
    uint32_t n_receive_elements,
    uint32_t n_samples,
    uint64_t n_output_voxels,
    uint32_t n_frames,
    uint32_t frame_stride,
    float f_number,
    float rx_start_s,
    float sampling_freq_hz,
    float sound_speed_m_s,
    float modulation_freq_hz,
    float tukey_alpha,
    InterpolationType interp_type,
    bool use_inverted_kernel
) {
#ifdef CUDA_PROFILE
    TIME_FUNCTION();
#endif

    // Check for potential overflow in sensor data indexing
    const uint64_t n_channel_data = static_cast<uint64_t>(n_receive_elements) * static_cast<uint64_t>(n_samples) * static_cast<uint64_t>(frame_stride);
    if (n_channel_data > UINT32_MAX) {
        throw std::runtime_error("Error: Sensor data array size exceeds 32-bit indexing limit. Maximum size is " +
                                std::to_string(UINT32_MAX) + " elements, but requested size is " +
                                std::to_string(n_channel_data) + " elements.");
    }

    if (tukey_alpha < 0.0f || tukey_alpha > 1.0f) {
        throw std::runtime_error("Error: tukey_alpha must be in range [0, 1], but got " +
                                std::to_string(tukey_alpha));
    }
    if (n_output_voxels > INT_MAX) {
        throw std::runtime_error("Error: Number of voxels (" + std::to_string(n_output_voxels) +
                                 ") exceeds the maximum integer value (" + std::to_string(INT_MAX) + ").");
    }
    bool apod_flag = tukey_alpha > 0.0f;
    const float inv_sound_speed_m_s = 1.0f / sound_speed_m_s;

    // Inverted-loop fast path (I/Q, nearest/linear, 16B-aligned frame_stride with
    // room for the trailing chunk); falls through to the original kernel otherwise.
    if (_try_beamform_inverted<DataType, StorageType>(
            d_channel_data, d_rx_coords_m, d_scan_coords_m, d_tx_arrivals_s, d_out,
            n_receive_elements, n_samples, n_output_voxels, n_frames, frame_stride,
            f_number, rx_start_s, sampling_freq_hz, inv_sound_speed_m_s,
            modulation_freq_hz, tukey_alpha, interp_type, use_inverted_kernel)) {
        return;
    }

    // Calculate block dimensions
    const int frames_per_block = min(n_frames, MAX_FRAME_THREADS_PER_BLOCK);
    const int voxels_per_block = calculate_voxels_per_block(frames_per_block);
    dim3 threads_per_block(frames_per_block, voxels_per_block);
    DEBUG_ASSERT(threads_per_block.x * threads_per_block.y <= 1024); // CUDA thread-count limit per block

    const int receive_elements_batch_size = calculate_receive_elements_batch_size(voxels_per_block);

#ifdef CUDA_DEBUG
    std::cout << "Thread dimensions: " << threads_per_block.x << " (frames) x " << threads_per_block.y
              << " (voxels) = " << threads_per_block.x * threads_per_block.y << " threads per block" << std::endl;
#endif

    // Calculate grid dimension - each block processes voxels_per_block voxels
    const int num_blocks = (n_output_voxels + voxels_per_block - 1) / voxels_per_block;
    const int max_blocks_per_dim = (1 << 16) - 32; // 2**16 - 32 is the max, CUDA recommends multiples of 32

    // Calculate grid dimensions ensuring x dimension doesn't exceed max_blocks_per_dim
    // x&y dimensions: voxel-batches, z dimension: receive-element-batches
    const int grid_x = min(max_blocks_per_dim, num_blocks);
    const int grid_y = (num_blocks + grid_x - 1) / grid_x;
    const int grid_z = (n_receive_elements + receive_elements_batch_size - 1) / receive_elements_batch_size;
    dim3 grid(grid_x, grid_y, grid_z);

#ifdef CUDA_PROFILE
    std::cout << "Grid dimensions: " << grid.x << " x " << grid.y << " x " << grid.z << " = "
              << grid.x * grid.y * grid.z << " blocks, each handling "
              << voxels_per_block << " voxels x " << receive_elements_batch_size
              << " receive_elements" << std::endl;
#endif

    // Check if our shared memory allocation will fit
    static constexpr int shared_mem_size = VOXELS_RECEIVE_ELEMENTS_BATCH_SIZE * sizeof(float2);
    int device_id;
    checkCudaErrors(cudaGetDevice(&device_id));
    int max_shared_mem;
    checkCudaErrors(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlock, device_id));

#ifdef CUDA_PROFILE
    std::cout << "Shared memory per block: " << shared_mem_size / 1024.0f << " KB" << std::endl;
    std::cout << "Maximum shared memory available: " << max_shared_mem / 1024.0f << " KB" << std::endl;
#endif

    if (shared_mem_size > max_shared_mem) {
        throw std::runtime_error("Error: Shared memory per block (" + std::to_string(shared_mem_size)
                  + " bytes) exceeds device limit (" + std::to_string(max_shared_mem)
                  + " bytes). Reducing DEFAULT_NUM_VOXELS_PER_BLOCK or DEFAULT_RECEIVE_ELEMENTS_BATCH_SIZE is required.");
    }

#ifdef CUDA_PROFILE
    cudaEvent_t start, stop;
    {
    TIME_SECTION("kernel_execution");

    // Time the kernel execution with native CUDA events
    // which only times the kernel execution, not the kernel launch
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));
#endif

    // Process all voxels with the kernel
    // We use template compile-time specialization to handle the different cases
    // Dispatch based on apodization and interpolation type
    if (apod_flag) {
        if (interp_type == InterpolationType::NearestNeighbor) {
            checkCudaErrors(cudaFuncSetCacheConfig(beamformKernel<DataType, true, InterpolationType::NearestNeighbor, StorageType>, CACHE_CONFIG));
            beamformKernel<DataType, true, InterpolationType::NearestNeighbor, StorageType><<<grid, threads_per_block>>>(
                d_channel_data, n_frames, frame_stride, n_receive_elements, n_samples,
                d_rx_coords_m, d_scan_coords_m, d_tx_arrivals_s, d_out,
                sampling_freq_hz, inv_sound_speed_m_s, modulation_freq_hz,
                f_number, tukey_alpha, rx_start_s, n_output_voxels, receive_elements_batch_size
            );
        } else if (interp_type == InterpolationType::Linear) {
            checkCudaErrors(cudaFuncSetCacheConfig(beamformKernel<DataType, true, InterpolationType::Linear, StorageType>, CACHE_CONFIG));
            beamformKernel<DataType, true, InterpolationType::Linear, StorageType><<<grid, threads_per_block>>>(
                d_channel_data, n_frames, frame_stride, n_receive_elements, n_samples,
                d_rx_coords_m, d_scan_coords_m, d_tx_arrivals_s, d_out,
                sampling_freq_hz, inv_sound_speed_m_s, modulation_freq_hz,
                f_number, tukey_alpha, rx_start_s, n_output_voxels, receive_elements_batch_size
            );
        } else { // Quadratic interpolation
            checkCudaErrors(cudaFuncSetCacheConfig(beamformKernel<DataType, true, InterpolationType::Quadratic, StorageType>, CACHE_CONFIG));
            beamformKernel<DataType, true, InterpolationType::Quadratic, StorageType><<<grid, threads_per_block>>>(
                d_channel_data, n_frames, frame_stride, n_receive_elements, n_samples,
                d_rx_coords_m, d_scan_coords_m, d_tx_arrivals_s, d_out,
                sampling_freq_hz, inv_sound_speed_m_s, modulation_freq_hz,
                f_number, tukey_alpha, rx_start_s, n_output_voxels, receive_elements_batch_size
            );
        }
    } else {
        if (interp_type == InterpolationType::NearestNeighbor) {
            checkCudaErrors(cudaFuncSetCacheConfig(beamformKernel<DataType, false, InterpolationType::NearestNeighbor, StorageType>, CACHE_CONFIG));
            beamformKernel<DataType, false, InterpolationType::NearestNeighbor, StorageType><<<grid, threads_per_block>>>(
                d_channel_data, n_frames, frame_stride, n_receive_elements, n_samples,
                d_rx_coords_m, d_scan_coords_m, d_tx_arrivals_s, d_out,
                sampling_freq_hz, inv_sound_speed_m_s, modulation_freq_hz,
                f_number, tukey_alpha, rx_start_s, n_output_voxels, receive_elements_batch_size
            );
        } else if (interp_type == InterpolationType::Linear) {
            checkCudaErrors(cudaFuncSetCacheConfig(beamformKernel<DataType, false, InterpolationType::Linear, StorageType>, CACHE_CONFIG));
            beamformKernel<DataType, false, InterpolationType::Linear, StorageType><<<grid, threads_per_block>>>(
                d_channel_data, n_frames, frame_stride, n_receive_elements, n_samples,
                d_rx_coords_m, d_scan_coords_m, d_tx_arrivals_s, d_out,
                sampling_freq_hz, inv_sound_speed_m_s, modulation_freq_hz,
                f_number, tukey_alpha, rx_start_s, n_output_voxels, receive_elements_batch_size
            );
        } else { // Quadratic interpolation
            checkCudaErrors(cudaFuncSetCacheConfig(beamformKernel<DataType, false, InterpolationType::Quadratic, StorageType>, CACHE_CONFIG));
            beamformKernel<DataType, false, InterpolationType::Quadratic, StorageType><<<grid, threads_per_block>>>(
                d_channel_data, n_frames, frame_stride, n_receive_elements, n_samples,
                d_rx_coords_m, d_scan_coords_m, d_tx_arrivals_s, d_out,
                sampling_freq_hz, inv_sound_speed_m_s, modulation_freq_hz,
                f_number, tukey_alpha, rx_start_s, n_output_voxels, receive_elements_batch_size
            );
        }
    }
    // Wait for kernel to complete
    checkCudaErrors(cudaDeviceSynchronize());

#ifdef CUDA_PROFILE
    } // End of kernel_execution section

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    // Calculate and print elapsed time
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Clean up timing events
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
#endif
}

/**
 * @brief Helper function to convert ndarray shape to string representation
 * @tparam ArrayType Any nanobind ndarray type
 * @param array The ndarray to get shape from
 * @return String representation of shape like "[dim0, dim1, dim2]"
 */
template<typename ArrayType>
std::string shape_to_string(const ArrayType& array) {
    std::string result = "[";
    for (size_t i = 0; i < array.ndim(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(array.shape(i));
    }
    result += "]";
    return result;
}

/**
 * @brief Check the dimensions of the input arrays
 */
template<typename SensorArrayType, typename CoordArrayType, typename TransmitArrayType, typename OutputArrayType>
void check_dimensions(
    const SensorArrayType& channel_data,
    const CoordArrayType& rx_coords_m,
    const CoordArrayType& scan_coords_m,
    const TransmitArrayType& tx_wave_arrivals_s,
    const OutputArrayType& out,
    size_t n_receive_elements,
    size_t n_samples,
    size_t n_output_voxels,
    size_t n_frames
) {
    // Validate dimensions
    if ((n_receive_elements != channel_data.shape(0)) || (n_receive_elements != rx_coords_m.shape(0))) {
        std::string error_msg = "Dimension mismatch in receive elements:\n";
        error_msg += "  Expected n_receive_elements: " + std::to_string(n_receive_elements) + "\n";
        error_msg += "  channel_data.shape: " + shape_to_string(channel_data) + "\n";
        error_msg += "  rx_coords_m.shape: " + shape_to_string(rx_coords_m) + "\n";
        error_msg += "→ channel_data.shape[0] and rx_coords_m.shape[0] must both equal n_receive_elements";
        throw std::runtime_error(error_msg);
    }
    if ((n_output_voxels != tx_wave_arrivals_s.shape(0)) || (n_output_voxels != scan_coords_m.shape(0)) || (n_output_voxels != out.shape(0))) {
        std::string error_msg = "Dimension mismatch in output voxels:\n";
        error_msg += "  Expected n_output_voxels: " + std::to_string(n_output_voxels) + "\n";
        error_msg += "  scan_coords_m.shape: " + shape_to_string(scan_coords_m) + "\n";
        error_msg += "  tx_wave_arrivals_s.shape: " + shape_to_string(tx_wave_arrivals_s) + "\n";
        error_msg += "  out.shape: " + shape_to_string(out) + "\n";
        error_msg += "→ scan_coords_m.shape[0], tx_wave_arrivals_s.shape[0], and out.shape[0] must all equal n_output_voxels";
        throw std::runtime_error(error_msg);
    }
    // channel_data may be allocated with more frames than are beamformed: the extra
    // frames pad the innermost stride so 128-bit loads stay available (see
    // _try_beamform_inverted). out.shape[1] is what decides how many are beamformed.
    if (n_frames != out.shape(1) || static_cast<int64_t>(n_frames) > channel_data.shape(2)) {
        std::string error_msg = "Dimension mismatch in frames:\n";
        error_msg += "  Expected n_frames: " + std::to_string(n_frames) + "\n";
        error_msg += "  channel_data.shape: " + shape_to_string(channel_data) + "\n";
        error_msg += "  out.shape: " + shape_to_string(out) + "\n";
        error_msg += "→ out.shape[1] must equal n_frames and channel_data.shape[2] must be at least n_frames";
        throw std::runtime_error(error_msg);
    }
}

/**
 * @brief Check device types of multiple arrays, validate supported devices, and issue performance warnings
 *
 * This function validates that all arrays are on supported devices (CPU or CUDA only),
 * determines the device distribution, and issues appropriate performance warnings.
 *
 * @tparam Arrays... Variadic array types
 * @param arrays... The arrays to check
 * @return int number of arrays on CPU
 * @throws std::runtime_error if any array is on an unsupported device
 */
template<typename... Arrays>
int check_devices(const Arrays&... arrays) {
    int cpu_count = 0;

    auto check_single_device = [&](const auto& array) {
        uint32_t device_type = array.device_type();

        // Validate supported device types
        if (device_type == nb::device::cpu::value) {
            cpu_count++;
        } else if (device_type == nb::device::cuda::value) {
            // do nothing
        } else {
            throw std::runtime_error(
                "Found input array on device: " + std::to_string(device_type) + ". Only CPU and CUDA devices are supported."
            );
        }
    };

    // Actually call the lambda function on each array
    (check_single_device(arrays), ...);

    return cpu_count;
}

/**
 * @brief Type aliases for thrust::allocate_unique to simplify complex template types
 */
template<typename T>
using device_allocator = thrust::device_allocator<T>;

template<typename T>
using device_unique_ptr = std::unique_ptr<
    T[],
    thrust::uninitialized_array_allocator_delete<
        T,
        typename thrust::detail::allocator_traits<device_allocator<T>>::template rebind_traits<T>::allocator_type
    >
>;

/**
 * @brief Main beamforming function that processes ultrasound data on the GPU
 *
 * This function automatically detects the device location of input arrays and handles
 * CPU<->GPU copying as needed.
 *
 *  This function implements delay-and-sum beamforming with the following features:
 * - Dynamic aperture growth based on F-number
 * - Cosine apodization with adjustable taper width
 * - Support for both RF and IQ data
 * - Multi-frame processing
 * - Configurable interpolation (nearest neighbor or linear)
 *
 *
 * @tparam DataType Either float (for RF data) or std::complex<float> (for I/Q data)
 * @param channel_data Input sensor data (I/Q or RF) [n_receive_elements, n_samples, frame_stride],
 *                     where frame_stride >= out.shape[1] (a longer innermost axis pads the stride)
 * @param rx_coords_m Receive element positions [n_receive_elements, 3] (in meters)
 * @param scan_coords_m Output voxel positions [n_output_voxels, 3] (in meters)
 * @param tx_wave_arrivals_s Transmit delays for each voxel [n_output_voxels] (in seconds)
 * @param out Output beamformed data [n_output_voxels, n_frames]
 * @param f_number F-number for aperture growth control
 * @param rx_start_s Receive start time offset (seconds, corresponds to t0 in biomecardio.com/publis/ultrasonics21.pdf)
 * @param sampling_freq_hz Sampling frequency (Hz)
 * @param sound_speed_m_s Speed of sound in medium (meters/second)
 * @param modulation_freq_hz Modulation frequency (Hz)
 * @param tukey_alpha Tukey window alpha for apodization (0=no apodization, 1=full apodization)
 * @param interp_type Interpolation method for sensor data sampling
 */
template<typename DataType>
void beamform(
    nb::ndarray<const DataType, nb::ndim<3>, nb::c_contig> channel_data,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> rx_coords_m,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan_coords_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> tx_wave_arrivals_s,
    nb::ndarray<DataType, nb::ndim<2>, nb::c_contig> out,
    float f_number,
    float rx_start_s,
    float sampling_freq_hz,
    float sound_speed_m_s,
    float modulation_freq_hz,
    float tukey_alpha,
    InterpolationType interp_type,
    bool use_inverted_kernel
) {
#ifdef CUDA_PROFILE
    TIME_FUNCTION();
#endif

    static_assert(std::is_same_v<DataType, float> || std::is_same_v<DataType, std::complex<float>>,
                  "DataType must be float (for RF data) or std::complex<float> (for I/Q data). "
                  "Other types like double/double2 or half/half2 would require kernel modifications.");

    // Extract dimensions from arrays
    size_t n_receive_elements = rx_coords_m.shape(0);
    size_t n_samples = channel_data.shape(1);
    size_t n_output_voxels = scan_coords_m.shape(0);
    size_t n_frames = out.shape(1);
    size_t frame_stride = channel_data.shape(2);

    check_dimensions(channel_data, rx_coords_m, scan_coords_m, tx_wave_arrivals_s, out,
        n_receive_elements, n_samples, n_output_voxels, n_frames);
    int cpu_count = check_devices(channel_data, rx_coords_m, scan_coords_m, tx_wave_arrivals_s, out);


    // If all arrays are already on CUDA, use the direct kernel call
    bool all_cuda = (cpu_count == 0);
    if (all_cuda) {
        // All arrays are on GPU - use direct kernel call
        const float3* d_rx_coords_m = reinterpret_cast<const float3*>(rx_coords_m.data());
        const float3* d_scan_coords_m = reinterpret_cast<const float3*>(scan_coords_m.data());
        const float* d_tx_arrivals_s = tx_wave_arrivals_s.data();

        if constexpr (std::is_same_v<DataType, std::complex<float>>) {
            const float2* d_channel_data = reinterpret_cast<const float2*>(channel_data.data());
            float2* d_out = reinterpret_cast<float2*>(out.data());
            _beamform_impl<float2>(d_channel_data, d_rx_coords_m, d_scan_coords_m, d_tx_arrivals_s, d_out,
                n_receive_elements, n_samples, n_output_voxels, n_frames, frame_stride,
                f_number, rx_start_s, sampling_freq_hz, sound_speed_m_s, modulation_freq_hz, tukey_alpha, interp_type,
                use_inverted_kernel);
        } else if constexpr (std::is_same_v<DataType, float>) {
            const float* d_channel_data = channel_data.data();
            float* d_out = out.data();
            _beamform_impl<float>(d_channel_data, d_rx_coords_m, d_scan_coords_m, d_tx_arrivals_s, d_out,
                n_receive_elements, n_samples, n_output_voxels, n_frames, frame_stride,
                f_number, rx_start_s, sampling_freq_hz, sound_speed_m_s, modulation_freq_hz, tukey_alpha, interp_type,
                use_inverted_kernel);
        }
        return;
    }

    // Use PyErr_WarnEx directly instead of Python warnings module
    // This is more reliable across different binding libraries
    std::string warning_msg = "Found " + std::to_string(cpu_count) + " input array(s) on CPU. " +
                              "This will add latency due to CPU<->GPU memory transfers. " +
                              "For optimal performance with CUDA beamforming, move arrays to GPU using cupy, jax, or similar.";
    if (PyErr_WarnEx(PyExc_UserWarning, warning_msg.c_str(), 1) < 0) {
        // Warning was converted to exception by warning filters - let it propagate
        // This respects Python's warning filter configuration (e.g., -W error)
        // https://docs.python.org/3/c-api/exceptions.html
        return;
    }

    // RAII-safe GPU device memory allocation for arrays that start on CPU
    // These unique_ptrs automatically call cudaFree when they go out of scope
    std::optional<device_unique_ptr<DataType>> d_unique_channel_data;
    std::optional<device_unique_ptr<float3>> d_unique_rx_coords_m;
    std::optional<device_unique_ptr<float3>> d_unique_scan_coords_m;
    std::optional<device_unique_ptr<float>> d_unique_tx_arrivals_s;
    std::optional<device_unique_ptr<DataType>> d_unique_out;

#ifdef CUDA_PROFILE
    {
        TIME_SECTION("allocate_gpu_memory_and_copy_cpu_arrays_to_gpu");
#endif
    // Allocate memory only for CPU arrays
    if (channel_data.device_type() == nb::device::cpu::value) {
        device_allocator<DataType> alloc;
        d_unique_channel_data = thrust::uninitialized_allocate_unique_n<DataType>(alloc, channel_data.size());
        checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(d_unique_channel_data->get()), channel_data.data(), channel_data.nbytes(), cudaMemcpyHostToDevice));
    }
    if (rx_coords_m.device_type() == nb::device::cpu::value) {
        device_allocator<float3> alloc;
        d_unique_rx_coords_m = thrust::uninitialized_allocate_unique_n<float3>(alloc, n_receive_elements);
        checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(d_unique_rx_coords_m->get()), rx_coords_m.data(), rx_coords_m.nbytes(), cudaMemcpyHostToDevice));
    }
    if (scan_coords_m.device_type() == nb::device::cpu::value) {
        device_allocator<float3> alloc;
        d_unique_scan_coords_m = thrust::uninitialized_allocate_unique_n<float3>(alloc, n_output_voxels);
        checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(d_unique_scan_coords_m->get()), scan_coords_m.data(), scan_coords_m.nbytes(), cudaMemcpyHostToDevice));
    }
    if (tx_wave_arrivals_s.device_type() == nb::device::cpu::value) {
        device_allocator<float> alloc;
        d_unique_tx_arrivals_s = thrust::uninitialized_allocate_unique_n<float>(alloc, tx_wave_arrivals_s.size());
        checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(d_unique_tx_arrivals_s->get()), tx_wave_arrivals_s.data(), tx_wave_arrivals_s.nbytes(), cudaMemcpyHostToDevice));
    }
    if (out.device_type() == nb::device::cpu::value) {
        device_allocator<DataType> alloc;
        d_unique_out = thrust::uninitialized_allocate_unique_n<DataType>(alloc, out.size());
        checkCudaErrors(cudaMemset(thrust::raw_pointer_cast(d_unique_out->get()), 0, out.nbytes()));
    }
#ifdef CUDA_PROFILE
    }
#endif

#ifdef CUDA_PROFILE
    {
        TIME_SECTION("call_beamform_impl");
#endif
    // If the array was manually copied to GPU device memory, use the device memory pointer
    // else use the nanobind array data, which was already on GPU device memory
    const DataType* d_channel_data = d_unique_channel_data ? thrust::raw_pointer_cast(d_unique_channel_data->get()) : channel_data.data();
    const float3* d_rx_coords_m = d_unique_rx_coords_m ? thrust::raw_pointer_cast(d_unique_rx_coords_m->get()) : reinterpret_cast<const float3*>(rx_coords_m.data());
    const float3* d_scan_coords_m = d_unique_scan_coords_m ? thrust::raw_pointer_cast(d_unique_scan_coords_m->get()) : reinterpret_cast<const float3*>(scan_coords_m.data());
    const float* d_tx_arrivals_s = d_unique_tx_arrivals_s ? thrust::raw_pointer_cast(d_unique_tx_arrivals_s->get()) : tx_wave_arrivals_s.data();
    DataType* d_out = d_unique_out ? thrust::raw_pointer_cast(d_unique_out->get()) : out.data();
    if constexpr (std::is_same_v<DataType, std::complex<float>>) {
        _beamform_impl<float2>(
            reinterpret_cast<const float2*>(d_channel_data),
            d_rx_coords_m,
            d_scan_coords_m,
            d_tx_arrivals_s,
            reinterpret_cast<float2*>(d_out),
            n_receive_elements,
            n_samples,
            n_output_voxels,
            n_frames,
            frame_stride,
            f_number,
            rx_start_s,
            sampling_freq_hz,
            sound_speed_m_s,
            modulation_freq_hz,
            tukey_alpha,
            interp_type,
            use_inverted_kernel
        );
    } else if constexpr (std::is_same_v<DataType, float>) {
        _beamform_impl<float>(
            reinterpret_cast<const float*>(d_channel_data),
            d_rx_coords_m,
            d_scan_coords_m,
            d_tx_arrivals_s,
            reinterpret_cast<float*>(d_out),
            n_receive_elements,
            n_samples,
            n_output_voxels,
            n_frames,
            frame_stride,
            f_number,
            rx_start_s,
            sampling_freq_hz,
            sound_speed_m_s,
            modulation_freq_hz,
            tukey_alpha,
            interp_type,
            use_inverted_kernel
        );
    }
#ifdef CUDA_PROFILE
    }
#endif

    // Copy results back to CPU if needed
#ifdef CUDA_PROFILE
    {
        TIME_SECTION("coy_gpu_result_to_cpu");
#endif
    if (out.device_type() == nb::device::cpu::value) {
        checkCudaErrors(cudaMemcpy(out.data(), d_out, out.nbytes(), cudaMemcpyDeviceToHost));
    }
#ifdef CUDA_PROFILE
    }
#endif
}

/**
 * @brief I/Q beamforming with FP16 (half2) channel-data storage.
 *
 * Identical math to beamform<std::complex<float>>: only the channel_data
 * STORAGE format changes, halving global-memory traffic and footprint.
 * Interpolation, apodization, phase rotation, and accumulation all stay FP32,
 * and `out` stays complex64 (so callers never need a complex32 dtype).
 *
 * channel_data holds the complex64 data converted to interleaved float16
 * (re, im) pairs: shape (n_rx, n_samples, 2*frame_stride), passed as a uint16
 * view. frame_stride may exceed out.shape[1] to pad the stride and keep the
 * inverted kernel's 128-bit loads. From cupy:
 *   half = iq.view(cp.float32).astype(cp.float16).view(cp.uint16)
 *
 * `out` is accumulated into with atomicAdd and must be zero-initialised by the
 * caller. float16 holds |x| <= 65504 with ~3 significant digits, so scale raw
 * ADC counts before converting. use_inverted_kernel=false forces the original
 * per-frame kernel (for A/B comparisons); see _try_beamform_inverted.
 *
 * Current restrictions: GPU arrays only (no CPU-copy path), I/Q only (an RF
 * __half variant is symmetric in the kernel template but not yet exposed).
 */
void beamform_fp16(
    nb::ndarray<const uint16_t, nb::ndim<3>, nb::c_contig> channel_data,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> rx_coords_m,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan_coords_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> tx_wave_arrivals_s,
    nb::ndarray<std::complex<float>, nb::ndim<2>, nb::c_contig> out,
    float f_number,
    float rx_start_s,
    float sampling_freq_hz,
    float sound_speed_m_s,
    float modulation_freq_hz,
    float tukey_alpha,
    InterpolationType interp_type,
    bool use_inverted_kernel
) {
    const size_t n_receive_elements = rx_coords_m.shape(0);
    const size_t n_samples = channel_data.shape(1);
    const size_t n_output_voxels = scan_coords_m.shape(0);
    const size_t n_frames = out.shape(1);

    // Interleaved (re, im) pairs, so the innermost axis holds 2 halves per frame.
    // It may be longer than 2*n_frames to pad the stride; see _try_beamform_inverted.
    if (channel_data.shape(0) != static_cast<int64_t>(n_receive_elements) ||
        channel_data.shape(2) % 2 != 0 ||
        channel_data.shape(2) < static_cast<int64_t>(2 * n_frames)) {
        throw std::runtime_error(
            "beamform_fp16: channel_data must have shape (n_rx, n_samples, 2*frame_stride) "
            "as uint16 (an interleaved-float16 view of the complex64 data) with "
            "frame_stride >= out.shape[1], got " +
            shape_to_string(channel_data) + " for out.shape " + shape_to_string(out));
    }
    const size_t frame_stride = channel_data.shape(2) / 2;
    if (tx_wave_arrivals_s.shape(0) != static_cast<int64_t>(n_output_voxels) ||
        out.shape(0) != static_cast<int64_t>(n_output_voxels)) {
        throw std::runtime_error("beamform_fp16: scan_coords_m, tx_wave_arrivals_s, and out "
                                 "must agree on n_output_voxels");
    }
    const int cpu_count = check_devices(channel_data, rx_coords_m, scan_coords_m,
                                        tx_wave_arrivals_s, out);
    if (cpu_count != 0) {
        throw std::runtime_error("beamform_fp16 requires all arrays on GPU "
                                 "(found " + std::to_string(cpu_count) + " CPU array(s))");
    }
    if (reinterpret_cast<std::uintptr_t>(channel_data.data()) % alignof(__half2) != 0) {
        throw std::runtime_error("beamform_fp16: channel_data must be 4-byte aligned "
                                 "(is it an odd-offset view of a uint16 buffer?)");
    }

    _beamform_impl<float2, __half2>(
        reinterpret_cast<const __half2*>(channel_data.data()),
        reinterpret_cast<const float3*>(rx_coords_m.data()),
        reinterpret_cast<const float3*>(scan_coords_m.data()),
        tx_wave_arrivals_s.data(),
        reinterpret_cast<float2*>(out.data()),
        n_receive_elements, n_samples, n_output_voxels, n_frames, frame_stride,
        f_number, rx_start_s, sampling_freq_hz, sound_speed_m_s,
        modulation_freq_hz, tukey_alpha, interp_type, use_inverted_kernel);
}

/**
 * @brief Python-facing vector-Jacobian product of beamform() for complex64 (I/Q) channel data.
 * GPU arrays only; every gradient buffer that is given is accumulated into and must be
 * zero-initialised by the caller. See beamformKernelInvIQVJP for the definitions.
 */
void beamform_vjp(
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> channel_data,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> rx_coords_m,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan_coords_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> tx_wave_arrivals_s,
    nb::ndarray<const std::complex<float>, nb::ndim<2>, nb::c_contig> grad_out,
    std::optional<nb::ndarray<std::complex<float>, nb::ndim<3>, nb::c_contig>> grad_channel_data,
    std::optional<nb::ndarray<float, nb::ndim<1>, nb::c_contig>> grad_tx_wave_arrivals_s,
    std::optional<nb::ndarray<float, nb::shape<-1, 3>, nb::c_contig>> grad_scan_coords_m,
    std::optional<nb::ndarray<float, nb::shape<-1, 3>, nb::c_contig>> grad_rx_coords_m,
    std::optional<nb::ndarray<double, nb::ndim<1>, nb::c_contig>> grad_sound_speed_m_s,
    std::optional<nb::ndarray<double, nb::ndim<1>, nb::c_contig>> grad_rx_start_s,
    float f_number,
    float rx_start_s,
    float sampling_freq_hz,
    float sound_speed_m_s,
    float modulation_freq_hz,
    float tukey_alpha,
    InterpolationType interp_type
) {
    const size_t n_receive_elements = rx_coords_m.shape(0);
    const size_t n_samples = channel_data.shape(1);
    const size_t frame_stride = channel_data.shape(2);
    const size_t n_output_voxels = scan_coords_m.shape(0);
    const size_t n_frames = grad_out.shape(1);

    if (channel_data.shape(0) != static_cast<int64_t>(n_receive_elements)) {
        throw std::runtime_error("beamform_vjp: channel_data.shape[0] must equal rx_coords_m.shape[0], got " +
                                 shape_to_string(channel_data) + " and " + shape_to_string(rx_coords_m));
    }
    if (tx_wave_arrivals_s.shape(0) != static_cast<int64_t>(n_output_voxels) ||
        grad_out.shape(0) != static_cast<int64_t>(n_output_voxels)) {
        throw std::runtime_error("beamform_vjp: scan_coords_m, tx_wave_arrivals_s and grad_out must agree on n_output_voxels");
    }
    if (n_frames > frame_stride) {
        throw std::runtime_error("beamform_vjp: grad_out.shape[1] (n_frames) must not exceed channel_data.shape[2]");
    }
    int cpu_count = check_devices(channel_data, rx_coords_m, scan_coords_m, tx_wave_arrivals_s, grad_out);
    auto check_grad = [&](const auto& opt, const char* name, auto expected_shape) {
        if (!opt) return;
        if (!expected_shape(*opt)) {
            throw std::runtime_error(std::string("beamform_vjp: ") + name + " has the wrong shape " + shape_to_string(*opt));
        }
        cpu_count += check_devices(*opt);
    };
    check_grad(grad_channel_data, "grad_channel_data", [&](const auto& a) {
        return a.shape(0) == channel_data.shape(0) && a.shape(1) == channel_data.shape(1) && a.shape(2) == channel_data.shape(2);
    });
    check_grad(grad_tx_wave_arrivals_s, "grad_tx_wave_arrivals_s", [&](const auto& a) { return a.shape(0) == static_cast<int64_t>(n_output_voxels); });
    check_grad(grad_scan_coords_m, "grad_scan_coords_m", [&](const auto& a) { return a.shape(0) == static_cast<int64_t>(n_output_voxels); });
    check_grad(grad_rx_coords_m, "grad_rx_coords_m", [&](const auto& a) { return a.shape(0) == static_cast<int64_t>(n_receive_elements); });
    check_grad(grad_sound_speed_m_s, "grad_sound_speed_m_s", [&](const auto& a) { return a.shape(0) == 1; });
    check_grad(grad_rx_start_s, "grad_rx_start_s", [&](const auto& a) { return a.shape(0) == 1; });
    if (cpu_count != 0) {
        throw std::runtime_error("beamform_vjp requires all arrays on GPU (found " + std::to_string(cpu_count) + " CPU array(s))");
    }

    _beamform_inverted_vjp<float2>(
        reinterpret_cast<const float2*>(channel_data.data()),
        reinterpret_cast<const float2*>(grad_out.data()),
        grad_channel_data ? reinterpret_cast<float2*>(grad_channel_data->data()) : nullptr,
        grad_tx_wave_arrivals_s ? grad_tx_wave_arrivals_s->data() : nullptr,
        grad_scan_coords_m ? reinterpret_cast<float3*>(grad_scan_coords_m->data()) : nullptr,
        grad_rx_coords_m ? reinterpret_cast<float3*>(grad_rx_coords_m->data()) : nullptr,
        grad_sound_speed_m_s ? grad_sound_speed_m_s->data() : nullptr,
        grad_rx_start_s ? grad_rx_start_s->data() : nullptr,
        reinterpret_cast<const float3*>(rx_coords_m.data()),
        reinterpret_cast<const float3*>(scan_coords_m.data()),
        tx_wave_arrivals_s.data(),
        n_receive_elements, n_samples, n_output_voxels, n_frames, frame_stride,
        f_number, rx_start_s, sampling_freq_hz, sound_speed_m_s, modulation_freq_hz, tukey_alpha, interp_type);
}

NB_MODULE(_cuda_impl, m) {
    m.doc() = "CUDA-accelerated ultrasound beamforming with nanobind";

    // Expose essential build-time version information
    m.attr("__nvcc_version__") = NVCC_VERSION_STR;

    // Perform compatibility checks at import time (warns if incompatible)
    checkCudaDriverCompatibility();
    checkComputeCapability();

    // Export InterpolationType enum to Python
    nb::enum_<InterpolationType>(m, "InterpolationType")
        .value("NearestNeighbor", InterpolationType::NearestNeighbor, "Use nearest neighbor interpolation (fastest)")
        .value("Linear", InterpolationType::Linear, "Use linear interpolation (default, good balance)")
        .value("Quadratic", InterpolationType::Quadratic, "Use quadratic interpolation (higher quality)")
        .export_values();

    // Overloaded GPU beamform functions - nanobind automatically handles dispatch based on argument types
    m.def("beamform", &beamform<std::complex<float>>,
        "channel_data"_a.noconvert(),
        "rx_coords_m"_a.noconvert(),
        "scan_coords_m"_a.noconvert(),
        "tx_wave_arrivals_s"_a.noconvert(),
        "out"_a.noconvert(),
        "f_number"_a,
        "rx_start_s"_a,
        "sampling_freq_hz"_a,
        "sound_speed_m_s"_a,
        "modulation_freq_hz"_a,
        "tukey_alpha"_a = 0.5f,
        "interp_type"_a = InterpolationType::Linear,
        "use_inverted_kernel"_a = true);

    m.def("beamform", &beamform<float>,
        "channel_data"_a.noconvert(),
        "rx_coords_m"_a.noconvert(),
        "scan_coords_m"_a.noconvert(),
        "tx_wave_arrivals_s"_a.noconvert(),
        "out"_a.noconvert(),
        "f_number"_a,
        "rx_start_s"_a,
        "sampling_freq_hz"_a,
        "sound_speed_m_s"_a,
        "modulation_freq_hz"_a = 0.0f,
        "tukey_alpha"_a = 0.5f,
        "interp_type"_a = InterpolationType::Linear,
        "use_inverted_kernel"_a = true);

    m.def("beamform_fp16", &beamform_fp16,
        "I/Q beamforming with FP16 (half2) channel-data storage (GPU arrays only).\n\n"
        "channel_data is the complex64 data as interleaved float16 (re, im) pairs viewed as "
        "uint16, shape (n_rx, n_samples, 2 * frame_stride) with frame_stride >= out.shape[1]: "
        "iq.view(cp.float32).astype(cp.float16).view(cp.uint16). "
        "Compute and out stay float32 / complex64; out is accumulated into and must be "
        "zero-initialised by the caller.",
        "channel_data"_a.noconvert(),
        "rx_coords_m"_a.noconvert(),
        "scan_coords_m"_a.noconvert(),
        "tx_wave_arrivals_s"_a.noconvert(),
        "out"_a.noconvert(),
        "f_number"_a,
        "rx_start_s"_a,
        "sampling_freq_hz"_a,
        "sound_speed_m_s"_a,
        "modulation_freq_hz"_a,
        "tukey_alpha"_a = 0.5f,
        "interp_type"_a = InterpolationType::Linear,
        "use_inverted_kernel"_a = true);

    m.def("beamform_vjp", &beamform_vjp,
        "Vector-Jacobian product (backward pass) of beamform() for complex64 channel data "
        "(GPU arrays only, inverted-kernel layout: nearest/linear interpolation, "
        "channel_data.shape[2] a multiple of 4 frames >= grad_out.shape[1]).\n\n"
        "grad_out is dL/dRe(out) + j dL/dIm(out). Each gradient buffer that is given is "
        "accumulated into and must be zero-initialised by the caller: grad_channel_data "
        "(the adjoint / backprojection), grad_tx_wave_arrivals_s, grad_scan_coords_m, "
        "grad_rx_coords_m, grad_sound_speed_m_s (float64, shape (1,)) and grad_rx_start_s "
        "(float64, shape (1,)). The aperture, sample bounds and apodization weight are "
        "treated as constants with respect to the geometry.",
        "channel_data"_a.noconvert(),
        "rx_coords_m"_a.noconvert(),
        "scan_coords_m"_a.noconvert(),
        "tx_wave_arrivals_s"_a.noconvert(),
        "grad_out"_a.noconvert(),
        "grad_channel_data"_a.noconvert() = nb::none(),
        "grad_tx_wave_arrivals_s"_a.noconvert() = nb::none(),
        "grad_scan_coords_m"_a.noconvert() = nb::none(),
        "grad_rx_coords_m"_a.noconvert() = nb::none(),
        "grad_sound_speed_m_s"_a.noconvert() = nb::none(),
        "grad_rx_start_s"_a.noconvert() = nb::none(),
        "f_number"_a,
        "rx_start_s"_a,
        "sampling_freq_hz"_a,
        "sound_speed_m_s"_a,
        "modulation_freq_hz"_a,
        "tukey_alpha"_a = 0.5f,
        "interp_type"_a = InterpolationType::Linear);
}
