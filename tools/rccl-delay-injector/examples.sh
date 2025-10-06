#!/bin/bash

# RCCL Delay Injector Example Scripts
# This script demonstrates various usage patterns for the delay injector

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if delay injector library exists
check_library() {
    if [ ! -f "libdelay_injector.so" ]; then
        print_error "libdelay_injector.so not found. Please build it first with 'make'"
        exit 1
    fi
    print_success "Delay injector library found"
}

# Example 1: Basic delay injection
example_basic() {
    print_status "Example 1: Basic delay injection"
    echo "Injecting 1000ns delay to AllReduce operations..."
    
    RCCL_DELAY_ALLREDUCE_NS=1000 \
    RCCL_DELAY_LOG=1 \
    LD_PRELOAD=./libdelay_injector.so \
    echo "This would run your RCCL program with AllReduce delays"
    
    print_success "Basic example completed"
    echo ""
}

# Example 2: Multiple collective delays
example_multiple() {
    print_status "Example 2: Multiple collective delays"
    echo "Setting different delays for different collectives..."
    
    RCCL_DELAY_ALLREDUCE_NS=1000 \
    RCCL_DELAY_BROADCAST_NS=500 \
    RCCL_DELAY_ALLGATHER_NS=2000 \
    RCCL_DELAY_REDUCESCATTER_NS=1500 \
    RCCL_DELAY_LOG=1 \
    LD_PRELOAD=./libdelay_injector.so \
    echo "This would run your RCCL program with multiple delays"
    
    print_success "Multiple delays example completed"
    echo ""
}

# Example 3: Group mode with warmup
example_group_mode() {
    print_status "Example 3: Group mode with warmup"
    echo "Using group-end mode with 10 operation warmup..."
    
    RCCL_DELAY_DEFAULT_NS=2000 \
    RCCL_DELAY_MODE=group_end \
    RCCL_DELAY_WARMUP=10 \
    RCCL_DELAY_LOG=2 \
    LD_PRELOAD=./libdelay_injector.so \
    echo "This would run your RCCL program in group mode"
    
    print_success "Group mode example completed"
    echo ""
}

# Example 4: Random jitter simulation
example_jitter() {
    print_status "Example 4: Random jitter simulation"
    echo "Simulating network latency with random jitter..."
    
    RCCL_DELAY_ALLREDUCE_NS=5000 \
    RCCL_DELAY_RANDOM=1 \
    RCCL_DELAY_JITTER_MAX_NS=1000 \
    RCCL_DELAY_LOG=1 \
    LD_PRELOAD=./libdelay_injector.so \
    echo "This would run your RCCL program with jitter simulation"
    
    print_success "Jitter simulation example completed"
    echo ""
}

# Example 5: Synchronous delays
example_sync() {
    print_status "Example 5: Synchronous delays"
    echo "Using synchronous delays for precise timing..."
    
    RCCL_DELAY_ALLREDUCE_NS=1000 \
    RCCL_DELAY_SYNC=1 \
    RCCL_DELAY_LOG=2 \
    LD_PRELOAD=./libdelay_injector.so \
    echo "This would run your RCCL program with sync delays"
    
    print_success "Synchronous delays example completed"
    echo ""
}

# Example 6: Performance testing configuration
example_performance() {
    print_status "Example 6: Performance testing configuration"
    echo "Configuration for performance testing with various delays..."
    
    cat << EOF
# Performance testing configuration
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
RCCL_DELAY_LOG=1 \
RCCL_DELAY_MODE=per_collective \
RCCL_DELAY_SYNC=0 \
RCCL_DELAY_RANDOM=1 \
RCCL_DELAY_JITTER_MAX_NS=100 \
LD_PRELOAD=./libdelay_injector.so your_performance_test
EOF
    
    print_success "Performance testing configuration shown"
    echo ""
}

# Example 7: Debugging configuration
example_debugging() {
    print_status "Example 7: Debugging configuration"
    echo "Configuration for debugging with verbose logging..."
    
    cat << EOF
# Debugging configuration
RCCL_DELAY_DEFAULT_NS=100 \
RCCL_DELAY_WARMUP=0 \
RCCL_DELAY_LOG=2 \
RCCL_DELAY_MODE=per_collective \
RCCL_DELAY_SYNC=0 \
LD_PRELOAD=./libdelay_injector.so your_debugging_program
EOF
    
    print_success "Debugging configuration shown"
    echo ""
}

# Show all examples
show_all_examples() {
    print_status "Running all delay injector examples..."
    echo ""
    
    example_basic
    example_multiple
    example_group_mode
    example_jitter
    example_sync
    example_performance
    example_debugging
    
    print_success "All examples completed!"
}

# Show usage
show_usage() {
    echo "RCCL Delay Injector Examples"
    echo ""
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  basic        - Basic delay injection example"
    echo "  multiple     - Multiple collective delays example"
    echo "  group        - Group mode with warmup example"
    echo "  jitter       - Random jitter simulation example"
    echo "  sync         - Synchronous delays example"
    echo "  performance  - Performance testing configuration"
    echo "  debugging    - Debugging configuration"
    echo "  all          - Show all examples (default)"
    echo "  help         - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 basic      # Show basic usage example"
    echo "  $0 all        # Show all examples"
    echo ""
}

# Main execution
main() {
    # Check if library exists
    check_library
    
    # Parse command line arguments
    case "${1:-all}" in
        basic)
            example_basic
            ;;
        multiple)
            example_multiple
            ;;
        group)
            example_group_mode
            ;;
        jitter)
            example_jitter
            ;;
        sync)
            example_sync
            ;;
        performance)
            example_performance
            ;;
        debugging)
            example_debugging
            ;;
        all)
            show_all_examples
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
