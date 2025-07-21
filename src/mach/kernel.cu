#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h> // float2 lerp

#include <iostream>
#include <cmath>
#include <complex>
#include <cstdio>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

// Add RAII support for CUDA memory
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
    Linear = 1            ///< Use linear interpolation (default)
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
 * @brief Template function for nearest neighbor interpolation with bounds checking
 * @tparam DataType Either float or float2
 * @param channel_data Pointer to sensor data
 * @param sample_idx Floating point sample index
 * @param receive_element_idx Index of receive element
 * @param frame_idx Index of frame
 * @param n_samples Number of samples per element
 * @param n_frames Number of frames
 * @param[out] is_valid Whether the sample is within bounds
 * @return Interpolated sensor sample (undefined if is_valid is false)
 */
template<typename DataType>
__device__ __forceinline__ DataType interpolate_nearest(
    const DataType* const __restrict__ channel_data,
    float sample_idx,
    uint32_t receive_element_idx,
    uint32_t frame_idx,
    uint32_t n_samples,
    uint32_t n_frames,
    bool& is_valid
) {
    // For nearest neighbor, check if rounded sample is in bounds
    if ((sample_idx < -0.5f) || (sample_idx > (n_samples - 0.5f))) {
        is_valid = false;
        return DataType{};  // Return default-constructed value (won't be used)
    }
    
    const unsigned int sample_idx_round = __float2uint_rn(sample_idx);  // Round to nearest
    DEBUG_ASSERT(sample_idx_round < n_samples);  // Verify sample index is in bounds
    const uint32_t channel_data_idx = receive_element_idx * n_samples * n_frames +
                                     sample_idx_round * n_frames +
                                     frame_idx;
    DEBUG_ASSERT(channel_data_idx < static_cast<uint64_t>(n_samples) * n_frames * (receive_element_idx + 1));  // Verify channel data index is in bounds  
    
    is_valid = true;
    return channel_data[channel_data_idx];
}

/**
 * @brief Template function for linear interpolation with bounds checking
 * @tparam DataType Either float or float2
 * @param channel_data Pointer to sensor data
 * @param sample_idx Floating point sample index
 * @param receive_element_idx Index of receive element
 * @param frame_idx Index of frame
 * @param n_samples Number of samples per element
 * @param n_frames Number of frames
 * @param[out] is_valid Whether the sample is within bounds
 * @return Interpolated sensor sample (undefined if is_valid is false)
 */
template<typename DataType>
__device__ __forceinline__ DataType interpolate_linear(
    const DataType* const __restrict__ channel_data,
    float sample_idx,
    uint32_t receive_element_idx,
    uint32_t frame_idx,
    uint32_t n_samples,
    uint32_t n_frames,
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
    
    const uint32_t channel_data_idx_floor = receive_element_idx * n_samples * n_frames +
                                           sample_idx_floor * n_frames +
                                           frame_idx;
    const uint32_t channel_data_idx_ceil = receive_element_idx * n_samples * n_frames +
                                          sample_idx_ceil * n_frames +
                                          frame_idx;
    
    DEBUG_ASSERT(channel_data_idx_floor < static_cast<uint64_t>(n_samples) * n_frames * (receive_element_idx + 1));  // Verify floor channel data index is in bounds
    DEBUG_ASSERT(channel_data_idx_ceil < static_cast<uint64_t>(n_samples) * n_frames * (receive_element_idx + 1));   // Verify ceil channel data index is in bounds
    
    is_valid = true;
    return lerp(channel_data[channel_data_idx_floor], channel_data[channel_data_idx_ceil], lerp_alpha);
}

/**
 * @brief Unified template function for interpolation dispatch with bounds checking
 * @tparam DataType Either float or float2
 * @tparam interpType Interpolation type (compile-time constant)
 * @param channel_data Pointer to sensor data
 * @param sample_idx Floating point sample index
 * @param receive_element_idx Index of receive element
 * @param frame_idx Index of frame
 * @param n_samples Number of samples per element
 * @param n_frames Number of frames
 * @param[out] is_valid Whether the sample is within bounds
 * @return Interpolated sensor sample (undefined if is_valid is false)
 */
template<typename DataType, InterpolationType interpType>
__device__ __forceinline__ DataType interpolate_sample(
    const DataType* const __restrict__ channel_data,
    float sample_idx,
    uint32_t receive_element_idx,
    uint32_t frame_idx,
    uint32_t n_samples,
    uint32_t n_frames,
    bool& is_valid
) {
    if constexpr (interpType == InterpolationType::NearestNeighbor) {
        return interpolate_nearest<DataType>(channel_data, sample_idx, receive_element_idx, frame_idx, n_samples, n_frames, is_valid);
    } else if constexpr (interpType == InterpolationType::Linear) {
        return interpolate_linear<DataType>(channel_data, sample_idx, receive_element_idx, frame_idx, n_samples, n_frames, is_valid);
    }
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
 * @tparam UseApodization Whether to apply Tukey window apodization
 * @tparam interpType Interpolation method for sensor data sampling
 * @param channel_data Input sensor data [n_receive_elements][n_samples][n_frames] (DataType)
 * @param n_frames Number of frames in channel_data (C-contiguous dimension)
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
template<typename DataType, bool UseApodization, InterpolationType interpType>
__global__ void beamformKernel(
    const DataType* const __restrict__ channel_data,
    __grid_constant__ const uint32_t n_frames,
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
    // Ensure DataType is one of the supported types for ultrasound beamforming
    static_assert(std::is_same_v<DataType, float> || std::is_same_v<DataType, float2>,
                  "DataType must be float (for RF data) or float2 (for I/Q data). "
                  "Other types like double/double2 or half/half2 would require kernel modifications.");
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
    const uint64_t n_channel_data = static_cast<uint64_t>(n_receive_elements) * static_cast<uint64_t>(n_samples) * static_cast<uint64_t>(n_frames);
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
            DataType sensor_sample = interpolate_sample<DataType, interpType>(
                channel_data, sample_idx, receive_element_idx, frame_idx, n_samples, n_frames, is_valid
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
 * @param d_channel_data Device pointer to sensor data [n_receive_elements, n_samples, n_frames]
 * @param d_rx_coords_m Device pointer to receive element positions [n_receive_elements, 3]
 * @param d_scan_coords_m Device pointer to output voxel positions [n_output_voxels, 3]
 * @param d_tx_arrivals_s Device pointer to transmit delays [n_output_voxels]
 * @param d_out Device pointer to output beamformed data [n_output_voxels, n_frames]
 * @param n_receive_elements Number of receive elements
 * @param n_samples Number of time samples
 * @param n_output_voxels Number of output voxels
 * @param n_frames Number of frames to beamform
 * @param f_number F-number for aperture growth control
 * @param rx_start_s Receive start time offset (seconds, corresponds to t0 in biomecardio.com/publis/ultrasonics21.pdf)
 * @param sampling_freq_hz Sampling frequency of channel_data (Hz)
 * @param sound_speed_m_s Speed of sound in medium (meters/second)
 * @param modulation_freq_hz Modulation frequency (Hz)
 * @param tukey_alpha Tukey window alpha for apodization (0=no apodization, 1=full apodization)
 * @param interp_type Interpolation method for sensor data sampling
 */
template<typename DataType>
void _beamform_impl(
    const DataType* d_channel_data,
    const float3* d_rx_coords_m,
    const float3* d_scan_coords_m,
    const float* d_tx_arrivals_s,
    DataType* d_out,
    uint32_t n_receive_elements,
    uint32_t n_samples,
    uint64_t n_output_voxels,
    uint32_t n_frames,
    float f_number,
    float rx_start_s,
    float sampling_freq_hz,
    float sound_speed_m_s,
    float modulation_freq_hz,
    float tukey_alpha,
    InterpolationType interp_type
) {
#ifdef CUDA_PROFILE
    TIME_FUNCTION();
#endif

    // Check for potential overflow in sensor data indexing
    const uint64_t n_channel_data = static_cast<uint64_t>(n_receive_elements) * static_cast<uint64_t>(n_samples) * static_cast<uint64_t>(n_frames);
    if (n_channel_data > UINT32_MAX) {
        throw std::runtime_error("Error: Sensor data array size exceeds 32-bit indexing limit. Maximum size is " +
                                std::to_string(UINT32_MAX) + " elements, but requested size is " +
                                std::to_string(n_channel_data) + " elements.");
    }

    if (tukey_alpha < 0.0f || tukey_alpha > 1.0f) {
        throw std::runtime_error("Error: tukey_alpha must be in range [0, 1], but got " +
                                std::to_string(tukey_alpha));
    }
    bool apod_flag = tukey_alpha > 0.0f;
    const float inv_sound_speed_m_s = 1.0f / sound_speed_m_s;

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
    if (n_output_voxels > INT_MAX) {
        throw std::runtime_error("Error: Number of voxels (" + std::to_string(n_output_voxels) +
                                 ") exceeds the maximum integer value (" + std::to_string(INT_MAX) + ").");
    }
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
            checkCudaErrors(cudaFuncSetCacheConfig(beamformKernel<DataType, true, InterpolationType::NearestNeighbor>, CACHE_CONFIG));
            beamformKernel<DataType, true, InterpolationType::NearestNeighbor><<<grid, threads_per_block>>>(
                d_channel_data, n_frames, n_receive_elements, n_samples,
                d_rx_coords_m, d_scan_coords_m, d_tx_arrivals_s, d_out,
                sampling_freq_hz, inv_sound_speed_m_s, modulation_freq_hz,
                f_number, tukey_alpha, rx_start_s, n_output_voxels, receive_elements_batch_size
            );
        } else { // Linear interpolation
            checkCudaErrors(cudaFuncSetCacheConfig(beamformKernel<DataType, true, InterpolationType::Linear>, CACHE_CONFIG));
            beamformKernel<DataType, true, InterpolationType::Linear><<<grid, threads_per_block>>>(
                d_channel_data, n_frames, n_receive_elements, n_samples,
                d_rx_coords_m, d_scan_coords_m, d_tx_arrivals_s, d_out,
                sampling_freq_hz, inv_sound_speed_m_s, modulation_freq_hz,
                f_number, tukey_alpha, rx_start_s, n_output_voxels, receive_elements_batch_size
            );
        }
    } else {
        if (interp_type == InterpolationType::NearestNeighbor) {
            checkCudaErrors(cudaFuncSetCacheConfig(beamformKernel<DataType, false, InterpolationType::NearestNeighbor>, CACHE_CONFIG));
            beamformKernel<DataType, false, InterpolationType::NearestNeighbor><<<grid, threads_per_block>>>(
                d_channel_data, n_frames, n_receive_elements, n_samples,
                d_rx_coords_m, d_scan_coords_m, d_tx_arrivals_s, d_out,
                sampling_freq_hz, inv_sound_speed_m_s, modulation_freq_hz,
                f_number, tukey_alpha, rx_start_s, n_output_voxels, receive_elements_batch_size
            );
        } else { // Linear interpolation
            checkCudaErrors(cudaFuncSetCacheConfig(beamformKernel<DataType, false, InterpolationType::Linear>, CACHE_CONFIG));
            beamformKernel<DataType, false, InterpolationType::Linear><<<grid, threads_per_block>>>(
                d_channel_data, n_frames, n_receive_elements, n_samples,
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
    if (n_frames != channel_data.shape(2) || n_frames != out.shape(1)) {
        std::string error_msg = "Dimension mismatch in frames:\n";
        error_msg += "  Expected n_frames: " + std::to_string(n_frames) + "\n";
        error_msg += "  channel_data.shape: " + shape_to_string(channel_data) + "\n";
        error_msg += "  out.shape: " + shape_to_string(out) + "\n";
        error_msg += "→ channel_data.shape[2] and out.shape[1] must both equal n_frames";
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
 * @param channel_data Input sensor data (I/Q or RF) [n_receive_elements, n_samples, n_frames]
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
    InterpolationType interp_type
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
                n_receive_elements, n_samples, n_output_voxels, n_frames,
                f_number, rx_start_s, sampling_freq_hz, sound_speed_m_s, modulation_freq_hz, tukey_alpha, interp_type);
        } else if constexpr (std::is_same_v<DataType, float>) {
            const float* d_channel_data = channel_data.data();
            float* d_out = out.data();
            _beamform_impl<float>(d_channel_data, d_rx_coords_m, d_scan_coords_m, d_tx_arrivals_s, d_out,
                n_receive_elements, n_samples, n_output_voxels, n_frames,
                f_number, rx_start_s, sampling_freq_hz, sound_speed_m_s, modulation_freq_hz, tukey_alpha, interp_type);
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
            f_number,
            rx_start_s,
            sampling_freq_hz,
            sound_speed_m_s,
            modulation_freq_hz,
            tukey_alpha,
            interp_type
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
            f_number,
            rx_start_s,
            sampling_freq_hz,
            sound_speed_m_s,
            modulation_freq_hz,
            tukey_alpha,
            interp_type
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
        .value("Linear", InterpolationType::Linear, "Use linear interpolation (default, higher quality)")
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
        "interp_type"_a = InterpolationType::Linear);

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
        "interp_type"_a = InterpolationType::Linear);
}
