/*************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * RCCL Deterministic Delay Injector
 * 
 * Injects deterministic delays into RCCL collective operations for testing
 * and debugging purposes.
 * 
 * Usage: LD_PRELOAD=./libdelay_injector.so your_rccl_program
 * 
 * Environment Variables:
 *   RCCL_DELAY_ALLREDUCE_NS     - Delay for AllReduce operations (nanoseconds)
 *   RCCL_DELAY_BROADCAST_NS     - Delay for Broadcast operations (nanoseconds)
 *   RCCL_DELAY_ALLGATHER_NS     - Delay for AllGather operations (nanoseconds)
 *   RCCL_DELAY_REDUCESCATTER_NS - Delay for ReduceScatter operations (nanoseconds)
 *   RCCL_DELAY_REDUCE_NS        - Delay for Reduce operations (nanoseconds)
 *   RCCL_DELAY_GATHER_NS        - Delay for Gather operations (nanoseconds)
 *   RCCL_DELAY_SCATTER_NS       - Delay for Scatter operations (nanoseconds)
 *   RCCL_DELAY_ALLTOALL_NS      - Delay for AllToAll operations (nanoseconds)
 *   RCCL_DELAY_ALLTOALLV_NS     - Delay for AllToAllV operations (nanoseconds)
 *   RCCL_DELAY_SEND_NS          - Delay for Send operations (nanoseconds)
 *   RCCL_DELAY_RECV_NS          - Delay for Recv operations (nanoseconds)
 *   RCCL_DELAY_DEFAULT_NS       - Default delay for unspecified operations (nanoseconds)
 *   RCCL_DELAY_WARMUP           - Number of operations to skip before applying delays
 *   RCCL_DELAY_LOG              - Enable logging (0=off, 1=basic, 2=verbose)
 *   RCCL_DELAY_MODE             - Delay mode: "per_collective" or "group_end"
 *   RCCL_DELAY_SYNC             - Synchronize after delay (0=async, 1=sync)
 *   RCCL_DELAY_RANDOM           - Add random jitter (0=off, 1=on)
 *   RCCL_DELAY_JITTER_MAX_NS    - Maximum random jitter in nanoseconds
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <random>
#include <mutex>
#include <algorithm>
#include <thread>
#include <functional>
#include <limits>
#include <climits>

namespace {

// Constants
constexpr unsigned long long MAX_DELAY_NS = 1000000000ULL;  // 1 second
constexpr unsigned long long MAX_JITTER_NS = 100000000ULL;  // 100ms
constexpr int MAX_WARMUP = 1000000;  // Prevent overflow

// Configuration structure
struct DelayConfig {
    unsigned long long delays[12];
    int warmup;
    int log_level;
    int sync_mode;
    int random_jitter;
    unsigned long long jitter_max_ns;
    bool per_collective_mode;
};

// Collective type enumeration
enum class CollectiveType : int {
    ALLREDUCE = 0,
    BROADCAST,
    ALLGATHER,
    REDUCESCATTER,
    REDUCE,
    GATHER,
    SCATTER,
    ALLTOALL,
    ALLTOALLV,
    SEND,
    RECV,
    DEFAULT,
    COUNT
};

// Global state
static std::atomic<bool> g_initialized{false};
static DelayConfig g_config{};
static std::atomic<int> g_total_calls{0};
static std::atomic<int> g_delayed_calls{0};
static std::atomic<int> g_error_count{0};
static std::mutex g_log_mutex;

// Thread-local state
static thread_local int t_calls = 0;
static thread_local int t_group_depth = 0;
static thread_local hipStream_t t_last_stream = nullptr;
static thread_local std::mt19937_64 t_rng;
static thread_local bool t_rng_initialized = false;

// Function pointer types
using ncclAllReduce_t = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, rcclComm_t, hipStream_t);
using ncclBroadcast_t = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, int, rcclComm_t, hipStream_t);
using ncclAllGather_t = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, rcclComm_t, hipStream_t);
using ncclReduceScatter_t = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, rcclComm_t, hipStream_t);
using ncclReduce_t = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, int, rcclComm_t, hipStream_t);
using ncclGather_t = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, int, rcclComm_t, hipStream_t);
using ncclScatter_t = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, int, rcclComm_t, hipStream_t);
using ncclAllToAll_t = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, rcclComm_t, hipStream_t);
using ncclAllToAllv_t = ncclResult_t (*)(const void*, void*, size_t*, size_t*, ncclDataType_t, rcclComm_t, hipStream_t);
using ncclSend_t = ncclResult_t (*)(const void*, size_t, ncclDataType_t, int, rcclComm_t, hipStream_t);
using ncclRecv_t = ncclResult_t (*)(void*, size_t, ncclDataType_t, int, rcclComm_t, hipStream_t);
using ncclGroupStart_t = ncclResult_t (*)(void);
using ncclGroupEnd_t = ncclResult_t (*)(void);

// Real RCCL function pointers
static ncclAllReduce_t real_ncclAllReduce = nullptr;
static ncclBroadcast_t real_ncclBroadcast = nullptr;
static ncclAllGather_t real_ncclAllGather = nullptr;
static ncclReduceScatter_t real_ncclReduceScatter = nullptr;
static ncclReduce_t real_ncclReduce = nullptr;
static ncclGather_t real_ncclGather = nullptr;
static ncclScatter_t real_ncclScatter = nullptr;
static ncclAllToAll_t real_ncclAllToAll = nullptr;
static ncclAllToAllv_t real_ncclAllToAllv = nullptr;
static ncclSend_t real_ncclSend = nullptr;
static ncclRecv_t real_ncclRecv = nullptr;
static ncclGroupStart_t real_ncclGroupStart = nullptr;
static ncclGroupEnd_t real_ncclGroupEnd = nullptr;

// Lookup tables
static const char* const collective_names[static_cast<int>(CollectiveType::COUNT)] = {
    "AllReduce", "Broadcast", "AllGather", "ReduceScatter", "Reduce",
    "Gather", "Scatter", "AllToAll", "AllToAllV", "Send", "Recv", "Default"
};

static const char* const env_var_names[static_cast<int>(CollectiveType::COUNT)] = {
    "RCCL_DELAY_ALLREDUCE_NS", "RCCL_DELAY_BROADCAST_NS", "RCCL_DELAY_ALLGATHER_NS",
    "RCCL_DELAY_REDUCESCATTER_NS", "RCCL_DELAY_REDUCE_NS", "RCCL_DELAY_GATHER_NS",
    "RCCL_DELAY_SCATTER_NS", "RCCL_DELAY_ALLTOALL_NS", "RCCL_DELAY_ALLTOALLV_NS",
    "RCCL_DELAY_SEND_NS", "RCCL_DELAY_RECV_NS", "RCCL_DELAY_DEFAULT_NS"
};

// GPU delay kernel
__global__ void delay_kernel(unsigned long long cycles) {
    const unsigned long long start = clock64();
    while ((clock64() - start) < cycles) {
        __asm__ __volatile__("" ::: "memory");
    }
}

// Convert nanoseconds to GPU cycles
static inline unsigned long long ns_to_cycles(unsigned long long ns) noexcept {
    if (ns == 0) return 0;
    
    int dev = 0;
    hipError_t err = hipGetDevice(&dev);
    if (err != hipSuccess) {
        return ns * 1000; // Fallback
    }
    
    hipDeviceProp_t prop;
    err = hipGetDeviceProperties(&prop, dev);
    if (err != hipSuccess) {
        return ns * 1000; // Fallback
    }
    
    const double cycles_per_ns = static_cast<double>(prop.clockRate) / 1e6;
    return static_cast<unsigned long long>(cycles_per_ns * static_cast<double>(ns));
}

// Initialize thread-local RNG safely
static inline void init_rng() noexcept {
    if (!t_rng_initialized) {
        try {
            t_rng.seed(std::random_device{}());
        } catch (...) {
            // Fallback to deterministic seed based on thread ID
            try {
                t_rng.seed(static_cast<unsigned int>(std::hash<std::thread::id>{}(std::this_thread::get_id())));
            } catch (...) {
                // Ultimate fallback to fixed seed
                t_rng.seed(0x12345678);
            }
        }
        t_rng_initialized = true;
    }
}

// Add random jitter if enabled
static inline unsigned long long add_jitter(unsigned long long base_ns) noexcept {
    if (!g_config.random_jitter || g_config.jitter_max_ns == 0) {
        return base_ns;
    }
    
    init_rng();
    
    const unsigned long long max_jitter = std::min(g_config.jitter_max_ns, MAX_JITTER_NS);
    std::uniform_int_distribution<unsigned long long> dist(0, max_jitter);
    const unsigned long long jitter = dist(t_rng);
    
    if (base_ns > MAX_DELAY_NS - jitter) {
        return MAX_DELAY_NS;
    }
    
    return base_ns + jitter;
}

// Apply deterministic delay
static inline void apply_delay(hipStream_t stream, unsigned long long delay_ns, CollectiveType coll_type) noexcept {
    if (delay_ns == 0) return;
    
    delay_ns = std::min(delay_ns, MAX_DELAY_NS);
    delay_ns = add_jitter(delay_ns);
    
    const unsigned long long cycles = ns_to_cycles(delay_ns);
    if (cycles == 0) return;
    
    const hipError_t err = hipLaunchKernel(delay_kernel, dim3(1), dim3(1), 0, stream, cycles);
    if (err != hipSuccess) {
        g_error_count.fetch_add(1);
        if (g_config.log_level > 0) {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            const char* error_msg = hipGetErrorString(err);
            fprintf(stderr, "[RCCL-DELAY] Error: Failed to launch delay kernel: %s\n", 
                    error_msg ? error_msg : "Unknown error");
        }
        return;
    }
    
    if (g_config.sync_mode) {
        const hipError_t sync_err = hipStreamSynchronize(stream);
        if (sync_err != hipSuccess) {
            g_error_count.fetch_add(1);
            if (g_config.log_level > 0) {
                std::lock_guard<std::mutex> lock(g_log_mutex);
                const char* error_msg = hipGetErrorString(sync_err);
                fprintf(stderr, "[RCCL-DELAY] Error: Failed to synchronize stream: %s\n", 
                        error_msg ? error_msg : "Unknown error");
            }
        }
    }
    
    if (g_config.log_level > 0) {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        const int coll_idx = static_cast<int>(coll_type);
        // Bounds check for array access
        if (coll_idx >= 0 && coll_idx < static_cast<int>(CollectiveType::COUNT)) {
            if (g_config.log_level == 1) {
                fprintf(stderr, "[RCCL-DELAY] +%lluns delay applied to %s\n", 
                        delay_ns, collective_names[coll_idx]);
            } else {
                fprintf(stderr, "[RCCL-DELAY] +%lluns delay applied to %s on stream %p (cycles: %llu)\n", 
                        delay_ns, collective_names[coll_idx], static_cast<void*>(stream), cycles);
            }
        }
    }
    
    g_delayed_calls.fetch_add(1);
}

// Check if delay should be applied
static inline bool should_apply_delay() noexcept {
    // Prevent overflow by clamping t_calls BEFORE comparison
    if (t_calls > MAX_WARMUP) {
        t_calls = MAX_WARMUP;
    }
    return t_calls >= g_config.warmup;
}

// Record stream for group mode
static inline void record_stream(hipStream_t stream) noexcept {
    t_last_stream = stream;
}

// Load symbol with error handling
template <typename FnPtr>
static bool load_symbol(FnPtr& fn, const char* name) noexcept {
    fn = reinterpret_cast<FnPtr>(dlsym(RTLD_NEXT, name));
    if (!fn) {
        // dlerror() is not thread-safe, but we're in constructor so it's safe here
        // Also check for null return from dlerror()
        const char* error_msg = dlerror();
        fprintf(stderr, "[RCCL-DELAY] Error: Failed to load symbol '%s': %s\n", 
                name, error_msg ? error_msg : "Unknown error");
        return false;
    }
    return true;
}

// Parse environment variable as unsigned long long
static unsigned long long parse_env_ull(const char* name, unsigned long long default_val) noexcept {
    const char* val = getenv(name);
    if (!val) return default_val;
    
    char* end;
    const unsigned long long result = strtoull(val, &end, 10);
    if (*end != '\0' || result == ULLONG_MAX || result > MAX_DELAY_NS) {
        fprintf(stderr, "[RCCL-DELAY] Warning: Invalid value for %s: '%s', using default %llu\n", 
                name, val, default_val);
        return default_val;
    }
    return result;
}

// Parse environment variable as integer
static int parse_env_int(const char* name, int default_val) noexcept {
    const char* val = getenv(name);
    if (!val) return default_val;
    
    char* end;
    const long result = strtol(val, &end, 10);
    if (*end != '\0' || result == LONG_MAX || result == LONG_MIN || 
        result < INT_MIN || result > INT_MAX) {
        fprintf(stderr, "[RCCL-DELAY] Warning: Invalid value for %s: '%s', using default %d\n", 
                name, val, default_val);
        return default_val;
    }
    return static_cast<int>(result);
}

// Initialize configuration from environment variables
static void init_config() noexcept {
    for (int i = 0; i < static_cast<int>(CollectiveType::COUNT); i++) {
        g_config.delays[i] = parse_env_ull(env_var_names[i], 0);
    }
    
    g_config.warmup = parse_env_int("RCCL_DELAY_WARMUP", 0);
    g_config.log_level = parse_env_int("RCCL_DELAY_LOG", 0);
    g_config.sync_mode = parse_env_int("RCCL_DELAY_SYNC", 0);
    g_config.random_jitter = parse_env_int("RCCL_DELAY_RANDOM", 0);
    g_config.jitter_max_ns = parse_env_ull("RCCL_DELAY_JITTER_MAX_NS", 0);
    
    // Clamp warmup to prevent overflow
    if (g_config.warmup > MAX_WARMUP) {
        g_config.warmup = MAX_WARMUP;
    }
    
    const char* mode = getenv("RCCL_DELAY_MODE");
    if (mode) {
        g_config.per_collective_mode = (strcmp(mode, "per_collective") == 0);
        if (!g_config.per_collective_mode && strcmp(mode, "group_end") != 0) {
            fprintf(stderr, "[RCCL-DELAY] Warning: Invalid mode '%s', using 'per_collective'\n", mode);
            g_config.per_collective_mode = true;
        }
    } else {
        g_config.per_collective_mode = true;
    }
}

// Print configuration summary
static void print_config() noexcept {
    bool has_delays = false;
    for (int i = 0; i < static_cast<int>(CollectiveType::COUNT); i++) {
        if (g_config.delays[i] > 0) {
            has_delays = true;
            break;
        }
    }
    
    if (!has_delays && g_config.log_level == 0) {
        return;
    }
    
    fprintf(stderr, "[RCCL-DELAY] Configuration:\n");
    fprintf(stderr, "[RCCL-DELAY]   Mode: %s\n", g_config.per_collective_mode ? "per_collective" : "group_end");
    fprintf(stderr, "[RCCL-DELAY]   Warmup: %d operations\n", g_config.warmup);
    fprintf(stderr, "[RCCL-DELAY]   Sync mode: %s\n", g_config.sync_mode ? "enabled" : "disabled");
    fprintf(stderr, "[RCCL-DELAY]   Random jitter: %s", g_config.random_jitter ? "enabled" : "disabled");
    if (g_config.random_jitter) {
        fprintf(stderr, " (max: %llu ns)", g_config.jitter_max_ns);
    }
    fprintf(stderr, "\n");
    
    fprintf(stderr, "[RCCL-DELAY]   Delays configured:\n");
    for (int i = 0; i < static_cast<int>(CollectiveType::COUNT); i++) {
        if (g_config.delays[i] > 0) {
            fprintf(stderr, "[RCCL-DELAY]     %s: %llu ns\n", collective_names[i], g_config.delays[i]);
        }
    }
    
    fprintf(stderr, "[RCCL-DELAY]   Log level: %d\n", g_config.log_level);
    fprintf(stderr, "[RCCL-DELAY] Ready to inject delays\n");
}

// Constructor - initialize everything
__attribute__((constructor))
static void init_delay_injector() {
    if (g_initialized.exchange(true)) return;
    
    fprintf(stderr, "[RCCL-DELAY] Initializing deterministic delay injector\n");
    
    init_config();
    
    bool success = true;
    success &= load_symbol(real_ncclAllReduce, "ncclAllReduce");
    success &= load_symbol(real_ncclBroadcast, "ncclBroadcast");
    success &= load_symbol(real_ncclAllGather, "ncclAllGather");
    success &= load_symbol(real_ncclReduceScatter, "ncclReduceScatter");
    success &= load_symbol(real_ncclReduce, "ncclReduce");
    success &= load_symbol(real_ncclGather, "ncclGather");
    success &= load_symbol(real_ncclScatter, "ncclScatter");
    success &= load_symbol(real_ncclAllToAll, "ncclAllToAll");
    success &= load_symbol(real_ncclAllToAllv, "ncclAllToAllv");
    success &= load_symbol(real_ncclSend, "ncclSend");
    success &= load_symbol(real_ncclRecv, "ncclRecv");
    success &= load_symbol(real_ncclGroupStart, "ncclGroupStart");
    success &= load_symbol(real_ncclGroupEnd, "ncclGroupEnd");
    
    if (!success) {
        fprintf(stderr, "[RCCL-DELAY] Error: Failed to load required RCCL symbols\n");
        abort();
    }
    
    print_config();
}

// Destructor - print statistics
__attribute__((destructor))
static void cleanup_delay_injector() {
    const int total = g_total_calls.load();
    const int delayed = g_delayed_calls.load();
    const int errors = g_error_count.load();
    
    if (total > 0) {
        fprintf(stderr, "[RCCL-DELAY] Statistics: %d total calls, %d delayed calls, %d errors\n", 
                total, delayed, errors);
    }
}

// Generic wrapper for collective functions
static inline ncclResult_t handle_collective_result(ncclResult_t result, hipStream_t stream, CollectiveType coll_type) noexcept {
    record_stream(stream);
    
    if (result == ncclSuccess && should_apply_delay()) {
        if (g_config.per_collective_mode) {
            apply_delay(stream, g_config.delays[static_cast<int>(coll_type)], coll_type);
        }
    }
    
    return result;
}

// Validate that all function pointers are loaded
static inline bool validate_function_pointers() noexcept {
    return real_ncclAllReduce && real_ncclBroadcast && real_ncclAllGather &&
           real_ncclReduceScatter && real_ncclReduce && real_ncclGather &&
           real_ncclScatter && real_ncclAllToAll && real_ncclAllToAllv &&
           real_ncclSend && real_ncclRecv && real_ncclGroupStart && real_ncclGroupEnd;
}

} // anonymous namespace

extern "C" {

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                          ncclDataType_t datatype, ncclRedOp_t op,
                          rcclComm_t comm, hipStream_t stream) {
    if (!validate_function_pointers()) {
        return ncclInvalidArgument;
    }
    
    g_total_calls.fetch_add(1);
    t_calls++;
    
    const ncclResult_t result = real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
    return handle_collective_result(result, stream, CollectiveType::ALLREDUCE);
}

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                          ncclDataType_t datatype, int root,
                          rcclComm_t comm, hipStream_t stream) {
    if (!validate_function_pointers()) {
        return ncclInvalidArgument;
    }
    
    g_total_calls.fetch_add(1);
    t_calls++;
    
    const ncclResult_t result = real_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
    return handle_collective_result(result, stream, CollectiveType::BROADCAST);
}

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t count,
                          ncclDataType_t datatype, rcclComm_t comm, hipStream_t stream) {
    if (!validate_function_pointers()) {
        return ncclInvalidArgument;
    }
    
    g_total_calls.fetch_add(1);
    t_calls++;
    
    const ncclResult_t result = real_ncclAllGather(sendbuff, recvbuff, count, datatype, comm, stream);
    return handle_collective_result(result, stream, CollectiveType::ALLGATHER);
}

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op,
                               rcclComm_t comm, hipStream_t stream) {
    if (!validate_function_pointers()) {
        return ncclInvalidArgument;
    }
    
    g_total_calls.fetch_add(1);
    t_calls++;
    
    const ncclResult_t result = real_ncclReduceScatter(sendbuff, recvbuff, count, datatype, op, comm, stream);
    return handle_collective_result(result, stream, CollectiveType::REDUCESCATTER);
}

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        rcclComm_t comm, hipStream_t stream) {
    if (!validate_function_pointers()) {
        return ncclInvalidArgument;
    }
    
    g_total_calls.fetch_add(1);
    t_calls++;
    
    const ncclResult_t result = real_ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
    return handle_collective_result(result, stream, CollectiveType::REDUCE);
}

ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t count,
                        ncclDataType_t datatype, int root,
                        rcclComm_t comm, hipStream_t stream) {
    if (!validate_function_pointers()) {
        return ncclInvalidArgument;
    }
    
    g_total_calls.fetch_add(1);
    t_calls++;
    
    const ncclResult_t result = real_ncclGather(sendbuff, recvbuff, count, datatype, root, comm, stream);
    return handle_collective_result(result, stream, CollectiveType::GATHER);
}

ncclResult_t ncclScatter(const void* sendbuff, void* recvbuff, size_t count,
                         ncclDataType_t datatype, int root,
                         rcclComm_t comm, hipStream_t stream) {
    if (!validate_function_pointers()) {
        return ncclInvalidArgument;
    }
    
    g_total_calls.fetch_add(1);
    t_calls++;
    
    const ncclResult_t result = real_ncclScatter(sendbuff, recvbuff, count, datatype, root, comm, stream);
    return handle_collective_result(result, stream, CollectiveType::SCATTER);
}

ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
                          ncclDataType_t datatype, rcclComm_t comm, hipStream_t stream) {
    if (!validate_function_pointers()) {
        return ncclInvalidArgument;
    }
    
    g_total_calls.fetch_add(1);
    t_calls++;
    
    const ncclResult_t result = real_ncclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
    return handle_collective_result(result, stream, CollectiveType::ALLTOALL);
}

ncclResult_t ncclAllToAllv(const void* sendbuff, void* recvbuff, size_t* sendcounts,
                           size_t* sdispls, size_t* recvcounts, size_t* rdispls,
                           ncclDataType_t datatype, rcclComm_t comm, hipStream_t stream) {
    if (!validate_function_pointers()) {
        return ncclInvalidArgument;
    }
    
    g_total_calls.fetch_add(1);
    t_calls++;
    
    const ncclResult_t result = real_ncclAllToAllv(sendbuff, recvbuff, sendcounts, sdispls, 
                                                  recvcounts, rdispls, datatype, comm, stream);
    return handle_collective_result(result, stream, CollectiveType::ALLTOALLV);
}

ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                      int peer, rcclComm_t comm, hipStream_t stream) {
    if (!validate_function_pointers()) {
        return ncclInvalidArgument;
    }
    
    g_total_calls.fetch_add(1);
    t_calls++;
    
    const ncclResult_t result = real_ncclSend(sendbuff, count, datatype, peer, comm, stream);
    return handle_collective_result(result, stream, CollectiveType::SEND);
}

ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, rcclComm_t comm, hipStream_t stream) {
    if (!validate_function_pointers()) {
        return ncclInvalidArgument;
    }
    
    g_total_calls.fetch_add(1);
    t_calls++;
    
    const ncclResult_t result = real_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
    return handle_collective_result(result, stream, CollectiveType::RECV);
}

ncclResult_t ncclGroupStart(void) {
    t_group_depth++;
    if (real_ncclGroupStart) {
        return real_ncclGroupStart();
    }
    return ncclInvalidArgument;
}

ncclResult_t ncclGroupEnd(void) {
    ncclResult_t result = ncclInvalidArgument;
    if (real_ncclGroupEnd) {
        result = real_ncclGroupEnd();
    }
    
    // Prevent underflow
    if (t_group_depth > 0) {
        t_group_depth--;
    }
    
    if (result == ncclSuccess && !g_config.per_collective_mode && t_last_stream != nullptr) {
        if (should_apply_delay()) {
            apply_delay(t_last_stream, g_config.delays[static_cast<int>(CollectiveType::DEFAULT)], CollectiveType::DEFAULT);
        }
    }
    
    return result;
}

} // extern "C"