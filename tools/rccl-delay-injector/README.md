# RCCL Deterministic Delay Injector

A comprehensive tool for injecting deterministic delays into RCCL collective operations for testing, debugging, and performance analysis.

## Features

- **Per-collective delay configuration**: Set different delays for each collective operation type
- **Group semantics support**: Delay at individual collectives or at group end
- **Random jitter**: Add configurable random jitter to delays
- **Comprehensive logging**: Multiple log levels for debugging
- **Thread-safe**: Proper handling of multi-threaded applications
- **Non-intrusive**: Uses LD_PRELOAD, no RCCL source modifications required
- **Production warnings**: Clear indicators this is for testing only
- **Robust error handling**: Graceful fallbacks and comprehensive validation
- **High performance**: Minimal overhead when delays are disabled
- **Clean implementation**: Well-documented and maintainable code

## Building

### Quick Build
```bash
# Build the delay injector
make

# Or manually compile
g++ -O2 -fPIC -shared -std=c++17 -Wall -Wextra -Wpedantic \
    -o libdelay_injector.so delay_injector.cpp -ldl -lhip_hcc
```

### Advanced Build Options
```bash
# Debug build with sanitizers
make debug CXXFLAGS+="-fsanitize=address -fsanitize=undefined"

# Optimized release build
make release

# Custom compiler
make CXX=clang++

# Check build
make check
```

### Installation
```bash
# Install to system
make install

# Custom prefix
make install PREFIX=/opt/rccl

# Uninstall
make uninstall
```

## Usage

### Basic Usage
```bash
# Inject 1000ns delay to all AllReduce operations
RCCL_DELAY_ALLREDUCE_NS=1000 LD_PRELOAD=./libdelay_injector.so your_rccl_program

# Inject different delays for different collectives
RCCL_DELAY_ALLREDUCE_NS=1000 \
RCCL_DELAY_BROADCAST_NS=500 \
RCCL_DELAY_ALLGATHER_NS=2000 \
LD_PRELOAD=./libdelay_injector.so your_rccl_program
```

### Advanced Configuration
```bash
# Complete configuration example
RCCL_DELAY_ALLREDUCE_NS=1000 \
RCCL_DELAY_BROADCAST_NS=500 \
RCCL_DELAY_ALLGATHER_NS=2000 \
RCCL_DELAY_REDUCESCATTER_NS=1500 \
RCCL_DELAY_REDUCE_NS=800 \
RCCL_DELAY_GATHER_NS=1200 \
RCCL_DELAY_SCATTER_NS=1200 \
RCCL_DELAY_ALLTOALL_NS=3000 \
RCCL_DELAY_ALLTOALLV_NS=3000 \
RCCL_DELAY_SEND_NS=400 \
RCCL_DELAY_RECV_NS=400 \
RCCL_DELAY_DEFAULT_NS=1000 \
RCCL_DELAY_WARMUP=10 \
RCCL_DELAY_LOG=2 \
RCCL_DELAY_MODE=per_collective \
RCCL_DELAY_SYNC=0 \
RCCL_DELAY_RANDOM=1 \
RCCL_DELAY_JITTER_MAX_NS=100 \
LD_PRELOAD=./libdelay_injector.so your_rccl_program
```

### Configuration Files
```bash
# Use pre-configured setups
source configs/performance_testing.env
LD_PRELOAD=./libdelay_injector.so your_program

# Or export and run
export $(cat configs/debugging.env | xargs)
LD_PRELOAD=./libdelay_injector.so your_program
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RCCL_DELAY_ALLREDUCE_NS` | Delay for AllReduce operations (nanoseconds) | 0 |
| `RCCL_DELAY_BROADCAST_NS` | Delay for Broadcast operations (nanoseconds) | 0 |
| `RCCL_DELAY_ALLGATHER_NS` | Delay for AllGather operations (nanoseconds) | 0 |
| `RCCL_DELAY_REDUCESCATTER_NS` | Delay for ReduceScatter operations (nanoseconds) | 0 |
| `RCCL_DELAY_REDUCE_NS` | Delay for Reduce operations (nanoseconds) | 0 |
| `RCCL_DELAY_GATHER_NS` | Delay for Gather operations (nanoseconds) | 0 |
| `RCCL_DELAY_SCATTER_NS` | Delay for Scatter operations (nanoseconds) | 0 |
| `RCCL_DELAY_ALLTOALL_NS` | Delay for AllToAll operations (nanoseconds) | 0 |
| `RCCL_DELAY_ALLTOALLV_NS` | Delay for AllToAllV operations (nanoseconds) | 0 |
| `RCCL_DELAY_SEND_NS` | Delay for Send operations (nanoseconds) | 0 |
| `RCCL_DELAY_RECV_NS` | Delay for Recv operations (nanoseconds) | 0 |
| `RCCL_DELAY_DEFAULT_NS` | Default delay for unspecified operations (nanoseconds) | 0 |
| `RCCL_DELAY_WARMUP` | Number of operations to skip before applying delays | 0 |
| `RCCL_DELAY_LOG` | Log level (0=off, 1=basic, 2=verbose) | 0 |
| `RCCL_DELAY_MODE` | Delay mode: "per_collective" or "group_end" | "per_collective" |
| `RCCL_DELAY_SYNC` | Synchronize after delay (0=async, 1=sync) | 0 |
| `RCCL_DELAY_RANDOM` | Add random jitter (0=off, 1=on) | 0 |
| `RCCL_DELAY_JITTER_MAX_NS` | Maximum random jitter in nanoseconds | 0 |

## Delay Modes

### Per-Collective Mode (default)
Delays are applied immediately after each collective operation completes. This is useful for:
- Testing individual collective performance
- Simulating network latency per operation
- Debugging specific collective issues

### Group-End Mode
Delays are applied only at `ncclGroupEnd()`, after all batched collectives complete. This is useful for:
- Testing group semantics
- Simulating batch processing delays
- Testing synchronization patterns

## Use Cases

### 1. Performance Testing
```bash
# Simulate network latency variations
RCCL_DELAY_ALLREDUCE_NS=5000 \
RCCL_DELAY_RANDOM=1 \
RCCL_DELAY_JITTER_MAX_NS=1000 \
LD_PRELOAD=./libdelay_injector.so your_performance_test
```

### 2. Debugging Race Conditions
```bash
# Add small delays to expose timing issues
RCCL_DELAY_DEFAULT_NS=100 \
RCCL_DELAY_LOG=2 \
LD_PRELOAD=./libdelay_injector.so your_debugging_test
```

### 3. Load Testing
```bash
# Simulate slower collective operations
RCCL_DELAY_ALLREDUCE_NS=10000 \
RCCL_DELAY_ALLGATHER_NS=15000 \
RCCL_DELAY_SYNC=1 \
LD_PRELOAD=./libdelay_injector.so your_load_test
```

### 4. Algorithm Validation
```bash
# Test different algorithms under artificial delays
RCCL_DELAY_ALLREDUCE_NS=2000 \
RCCL_DELAY_MODE=group_end \
LD_PRELOAD=./libdelay_injector.so your_algorithm_test
```

## Implementation Details

### GPU-Based Timing
The delay injector uses GPU `clock64()` for precise, deterministic delays. This ensures:
- Accurate timing regardless of CPU scheduling
- Deterministic behavior across runs
- No interference with CPU-side timing

### Thread Safety
- Thread-local state for per-thread call counting
- Atomic counters for global statistics
- Mutex-protected logging
- Proper handling of group semantics across threads

### Error Handling
- Graceful fallback for missing symbols
- Robust environment variable parsing
- HIP error checking and reporting
- Comprehensive validation of configuration

## Limitations

1. **Testing Only**: This tool is designed for testing and debugging only. Do not use in production.
2. **HIP Dependency**: Requires HIP runtime for GPU timing
3. **Symbol Loading**: Requires RCCL symbols to be available at runtime
4. **Stream Dependency**: Delays are applied on the same stream as the collective

## Troubleshooting

### Common Issues

1. **Symbol Loading Errors**
   ```
   [RCCL-DELAY] Error: Failed to load symbol 'ncclAllReduce': undefined symbol
   ```
   **Solution**: Ensure RCCL is properly installed and in the library path.

2. **HIP Errors**
   ```
   [RCCL-DELAY] Error: Failed to launch delay kernel: hipErrorInvalidDevice
   ```
   **Solution**: Ensure HIP is properly initialized before RCCL operations.

3. **No Delays Applied**
   ```
   [RCCL-DELAY] Statistics: 100 total calls, 0 delayed calls
   ```
   **Solution**: Check that delay values are set and warmup period has passed.

### Debug Mode

Enable verbose logging to debug issues:
```bash
RCCL_DELAY_LOG=2 LD_PRELOAD=./libdelay_injector.so your_program
```

## Contributing

When contributing to this tool:
1. Maintain backward compatibility with existing environment variables
2. Add comprehensive error handling
3. Update documentation for new features
4. Test with various RCCL applications
5. Follow the existing code style and patterns

## License

This tool follows the same license as RCCL. See LICENSE.txt for details.
