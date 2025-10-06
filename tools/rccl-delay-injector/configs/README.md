# RCCL Delay Injector Configuration Examples

This directory contains example configurations for different use cases of the RCCL Delay Injector.

## Configuration Files

### `performance_testing.env`
Configuration for performance testing with realistic network delays:
```bash
# Performance testing configuration
RCCL_DELAY_ALLREDUCE_NS=1000
RCCL_DELAY_BROADCAST_NS=500
RCCL_DELAY_ALLGATHER_NS=2000
RCCL_DELAY_REDUCESCATTER_NS=1500
RCCL_DELAY_REDUCE_NS=800
RCCL_DELAY_GATHER_NS=1200
RCCL_DELAY_SCATTER_NS=1200
RCCL_DELAY_ALLTOALL_NS=3000
RCCL_DELAY_ALLTOALLV_NS=3000
RCCL_DELAY_SEND_NS=400
RCCL_DELAY_RECV_NS=400
RCCL_DELAY_DEFAULT_NS=1000
RCCL_DELAY_WARMUP=10
RCCL_DELAY_LOG=1
RCCL_DELAY_MODE=per_collective
RCCL_DELAY_SYNC=0
RCCL_DELAY_RANDOM=1
RCCL_DELAY_JITTER_MAX_NS=100
```

### `debugging.env`
Configuration for debugging with minimal delays and verbose logging:
```bash
# Debugging configuration
RCCL_DELAY_DEFAULT_NS=100
RCCL_DELAY_WARMUP=0
RCCL_DELAY_LOG=2
RCCL_DELAY_MODE=per_collective
RCCL_DELAY_SYNC=0
RCCL_DELAY_RANDOM=0
RCCL_DELAY_JITTER_MAX_NS=0
```

### `load_testing.env`
Configuration for load testing with significant delays:
```bash
# Load testing configuration
RCCL_DELAY_ALLREDUCE_NS=10000
RCCL_DELAY_BROADCAST_NS=5000
RCCL_DELAY_ALLGATHER_NS=15000
RCCL_DELAY_REDUCESCATTER_NS=12000
RCCL_DELAY_REDUCE_NS=8000
RCCL_DELAY_GATHER_NS=10000
RCCL_DELAY_SCATTER_NS=10000
RCCL_DELAY_ALLTOALL_NS=20000
RCCL_DELAY_ALLTOALLV_NS=20000
RCCL_DELAY_SEND_NS=3000
RCCL_DELAY_RECV_NS=3000
RCCL_DELAY_DEFAULT_NS=10000
RCCL_DELAY_WARMUP=5
RCCL_DELAY_LOG=1
RCCL_DELAY_MODE=per_collective
RCCL_DELAY_SYNC=1
RCCL_DELAY_RANDOM=0
RCCL_DELAY_JITTER_MAX_NS=0
```

### `network_simulation.env`
Configuration for simulating network latency variations:
```bash
# Network simulation configuration
RCCL_DELAY_ALLREDUCE_NS=5000
RCCL_DELAY_BROADCAST_NS=2000
RCCL_DELAY_ALLGATHER_NS=8000
RCCL_DELAY_REDUCESCATTER_NS=6000
RCCL_DELAY_REDUCE_NS=3000
RCCL_DELAY_GATHER_NS=4000
RCCL_DELAY_SCATTER_NS=4000
RCCL_DELAY_ALLTOALL_NS=10000
RCCL_DELAY_ALLTOALLV_NS=10000
RCCL_DELAY_SEND_NS=1500
RCCL_DELAY_RECV_NS=1500
RCCL_DELAY_DEFAULT_NS=5000
RCCL_DELAY_WARMUP=20
RCCL_DELAY_LOG=1
RCCL_DELAY_MODE=per_collective
RCCL_DELAY_SYNC=0
RCCL_DELAY_RANDOM=1
RCCL_DELAY_JITTER_MAX_NS=1000
```

### `group_testing.env`
Configuration for testing group semantics:
```bash
# Group testing configuration
RCCL_DELAY_DEFAULT_NS=2000
RCCL_DELAY_WARMUP=0
RCCL_DELAY_LOG=2
RCCL_DELAY_MODE=group_end
RCCL_DELAY_SYNC=0
RCCL_DELAY_RANDOM=0
RCCL_DELAY_JITTER_MAX_NS=0
```

## Usage

### Using Configuration Files

```bash
# Load configuration from file
source performance_testing.env
LD_PRELOAD=./libdelay_injector.so your_rccl_program

# Or export variables and run
export $(cat performance_testing.env | xargs)
LD_PRELOAD=./libdelay_injector.so your_rccl_program
```

### Custom Configuration

Create your own configuration file:

```bash
# Create custom configuration
cat > my_config.env << EOF
RCCL_DELAY_ALLREDUCE_NS=1500
RCCL_DELAY_LOG=1
RCCL_DELAY_WARMUP=5
EOF

# Use custom configuration
source my_config.env
LD_PRELOAD=./libdelay_injector.so your_rccl_program
```

## Configuration Guidelines

### Delay Values
- **Small delays (100-1000ns)**: For debugging and fine-tuning
- **Medium delays (1000-10000ns)**: For performance testing
- **Large delays (10000+ns)**: For load testing and stress testing

### Warmup Period
- **0**: Apply delays immediately (good for debugging)
- **5-10**: Skip initial operations (good for performance testing)
- **20+**: Skip many operations (good for long-running tests)

### Log Levels
- **0**: No logging (production-like)
- **1**: Basic logging (shows delays applied)
- **2**: Verbose logging (shows detailed information)

### Modes
- **per_collective**: Apply delay after each collective (default)
- **group_end**: Apply delay only at ncclGroupEnd()

### Synchronization
- **0**: Asynchronous delays (faster, less precise)
- **1**: Synchronous delays (slower, more precise)

### Random Jitter
- **0**: No jitter (deterministic)
- **1**: Add random jitter (more realistic network simulation)

## Best Practices

1. **Start Small**: Begin with small delay values and increase gradually
2. **Test Incrementally**: Test one collective type at a time
3. **Use Appropriate Warmup**: Set warmup based on your test scenario
4. **Monitor Performance**: Use logging to verify delays are applied
5. **Clean Up**: Remove delay injector for production runs

## Troubleshooting

### Common Issues

1. **No Delays Applied**
   - Check that delay values are set (> 0)
   - Verify warmup period has passed
   - Ensure log level > 0 to see activity

2. **Unexpected Behavior**
   - Check for conflicting environment variables
   - Verify RCCL symbols are loaded correctly
   - Use verbose logging to debug

3. **Performance Impact**
   - Reduce delay values if too slow
   - Use asynchronous mode (SYNC=0)
   - Consider using group mode for batch operations
