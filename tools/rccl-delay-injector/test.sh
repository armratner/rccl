#!/bin/bash

# RCCL Delay Injector Test Script
# This script tests the delay injector with a simple RCCL program

set -euo pipefail

# Configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TEST_PROGRAM="test_delay_injector"
readonly TEST_SOURCE="test_delay_injector.cpp"
readonly DELAY_LIBRARY="libdelay_injector.so"

# Utility Functions
print_header() {
    echo "=========================================="
    echo "RCCL Delay Injector Test Suite"
    echo "=========================================="
    echo
}

print_status() {
    echo "[TEST] $1"
}

print_success() {
    echo "[PASS] $1"
}

print_error() {
    echo "[FAIL] $1"
}

print_warning() {
    echo "[WARN] $1"
}

print_info() {
    echo "[INFO] $1"
}

print_step() {
    echo "[STEP] $1"
}

# Dependency Checks
check_dependencies() {
    print_step "Checking dependencies..."
    
    # Check if g++ is available
    if ! command -v g++ &> /dev/null; then
        print_error "g++ compiler not found"
        print_info "Please install g++ compiler"
        exit 1
    fi
    print_success "g++ compiler found: $(g++ --version | head -n1)"
    
    # Check if delay injector library exists
    if [ ! -f "$DELAY_LIBRARY" ]; then
        print_error "$DELAY_LIBRARY not found"
        print_info "Please build it first with 'make'"
        exit 1
    fi
    print_success "Delay injector library found: $DELAY_LIBRARY"
    
    # Check if RCCL headers are available
    local rccl_found=false
    for path in "/opt/rocm/include" "/usr/include" "/usr/local/include"; do
        if [ -f "$path/rccl/rccl.h" ]; then
            print_success "RCCL headers found: $path/rccl/rccl.h"
            rccl_found=true
            break
        fi
    done
    
    if [ "$rccl_found" = false ]; then
        print_warning "RCCL headers not found in standard locations"
        print_info "Please ensure RCCL is installed and headers are accessible"
    fi
    
    # Check if HIP headers are available
    local hip_found=false
    for path in "/opt/rocm/include" "/usr/include" "/usr/local/include"; do
        if [ -f "$path/hip/hip_runtime.h" ]; then
            print_success "HIP headers found: $path/hip/hip_runtime.h"
            hip_found=true
            break
        fi
    done
    
    if [ "$hip_found" = false ]; then
        print_warning "HIP headers not found in standard locations"
        print_info "Please ensure HIP is installed and headers are accessible"
    fi
}

# Create test program
create_test_program() {
    print_status "Creating test program..."
    
    cat > "$TEST_SOURCE" << 'EOF'
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    std::cout << "RCCL Delay Injector Test Program" << std::endl;
    
    // Initialize HIP
    hipError_t hip_err = hipInit(0);
    if (hip_err != hipSuccess) {
        std::cerr << "HIP initialization failed: " << hipGetErrorString(hip_err) << std::endl;
        return 1;
    }
    
    int device_count;
    hip_err = hipGetDeviceCount(&device_count);
    if (hip_err != hipSuccess) {
        std::cerr << "Failed to get device count: " << hipGetErrorString(hip_err) << std::endl;
        return 1;
    }
    
    std::cout << "Found " << device_count << " HIP devices" << std::endl;
    
    if (device_count < 2) {
        std::cout << "Warning: Need at least 2 devices for multi-GPU test" << std::endl;
        std::cout << "Running single-device test instead..." << std::endl;
        
        // Single device test
        hipSetDevice(0);
        
        hipStream_t stream;
        hipStreamCreate(&stream);
        
        // Create a simple test
        const size_t N = 1024;
        float *h_data = new float[N];
        float *d_data;
        
        // Initialize host data
        for (size_t i = 0; i < N; i++) {
            h_data[i] = 1.0f;
        }
        
        // Allocate device memory
        hipMalloc(&d_data, N * sizeof(float));
        hipMemcpy(d_data, h_data, N * sizeof(float), hipMemcpyHostToDevice);
        
        std::cout << "Test data initialized" << std::endl;
        
        // Clean up
        hipFree(d_data);
        hipStreamDestroy(stream);
        delete[] h_data;
        
        std::cout << "Single-device test completed successfully" << std::endl;
        return 0;
    }
    
    // Multi-device test
    std::cout << "Running multi-device test..." << std::endl;
    
    // Initialize RCCL
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    
    ncclComm_t comm;
    int rank = 0;
    int nranks = std::min(device_count, 4); // Test with up to 4 devices
    
    ncclResult_t nccl_err = ncclCommInitRank(&comm, nranks, id, rank);
    if (nccl_err != ncclSuccess) {
        std::cerr << "RCCL initialization failed" << std::endl;
        return 1;
    }
    
    std::cout << "RCCL initialized with " << nranks << " ranks" << std::endl;
    
    // Set device
    hipSetDevice(rank);
    
    // Create stream
    hipStream_t stream;
    hipStreamCreate(&stream);
    
    // Test data
    const size_t N = 1024;
    float *h_data = new float[N];
    float *d_data;
    
    // Initialize host data
    for (size_t i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }
    
    // Allocate device memory
    hipMalloc(&d_data, N * sizeof(float));
    hipMemcpy(d_data, h_data, N * sizeof(float), hipMemcpyHostToDevice);
    
    std::cout << "Test data initialized" << std::endl;
    
    // Test AllReduce (this should trigger delay injection if configured)
    std::cout << "Testing AllReduce operation..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    nccl_err = ncclAllReduce(d_data, d_data, N, ncclFloat, ncclSum, comm, stream);
    if (nccl_err != ncclSuccess) {
        std::cerr << "AllReduce failed" << std::endl;
        ncclCommDestroy(comm);
        hipFree(d_data);
        hipStreamDestroy(stream);
        delete[] h_data;
        return 1;
    }
    
    hipStreamSynchronize(stream);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "AllReduce completed in " << duration.count() << " microseconds" << std::endl;
    
    // Test Broadcast
    std::cout << "Testing Broadcast operation..." << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    
    nccl_err = ncclBroadcast(d_data, d_data, N, ncclFloat, 0, comm, stream);
    if (nccl_err != ncclSuccess) {
        std::cerr << "Broadcast failed" << std::endl;
        ncclCommDestroy(comm);
        hipFree(d_data);
        hipStreamDestroy(stream);
        delete[] h_data;
        return 1;
    }
    
    hipStreamSynchronize(stream);
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Broadcast completed in " << duration.count() << " microseconds" << std::endl;
    
    // Clean up
    ncclCommDestroy(comm);
    hipFree(d_data);
    hipStreamDestroy(stream);
    delete[] h_data;
    
    std::cout << "Multi-device test completed successfully" << std::endl;
    return 0;
}
EOF
    
    print_success "Test program created: $TEST_SOURCE"
}

# Compile test program
compile_test_program() {
    print_status "Compiling test program..."
    
    # Try to find RCCL include path
    RCCL_INCLUDE=""
    if [ -d "/opt/rocm/include" ]; then
        RCCL_INCLUDE="-I/opt/rocm/include"
    elif [ -d "/usr/include" ]; then
        RCCL_INCLUDE="-I/usr/include"
    fi
    
    # Try to find HIP include path
    HIP_INCLUDE=""
    if [ -d "/opt/rocm/include" ]; then
        HIP_INCLUDE="-I/opt/rocm/include"
    elif [ -d "/usr/include" ]; then
        HIP_INCLUDE="-I/usr/include"
    fi
    
    # Compile
    g++ -std=c++11 -O2 -o "$TEST_PROGRAM" "$TEST_SOURCE" $RCCL_INCLUDE $HIP_INCLUDE -lrccl -lhip_hcc
    
    if [ $? -eq 0 ]; then
        print_success "Test program compiled successfully"
    else
        print_error "Failed to compile test program"
        exit 1
    fi
}

# Run test without delay injector
run_baseline_test() {
    print_status "Running baseline test (no delay injector)..."
    
    ./"$TEST_PROGRAM" > baseline_output.txt 2>&1
    
    if [ $? -eq 0 ]; then
        print_success "Baseline test passed"
    else
        print_error "Baseline test failed"
        cat baseline_output.txt
        return 1
    fi
}

# Run test with delay injector
run_delay_test() {
    print_status "Running test with delay injector..."
    
    # Test with AllReduce delay
    RCCL_DELAY_ALLREDUCE_NS=1000 \
    RCCL_DELAY_BROADCAST_NS=500 \
    RCCL_DELAY_LOG=1 \
    LD_PRELOAD=./"$DELAY_LIBRARY" \
    ./"$TEST_PROGRAM" > delay_output.txt 2>&1
    
    if [ $? -eq 0 ]; then
        print_success "Delay injector test passed"
    else
        print_error "Delay injector test failed"
        cat delay_output.txt
        return 1
    fi
}

# Compare results
compare_results() {
    print_status "Comparing test results..."
    
    if [ -f "baseline_output.txt" ] && [ -f "delay_output.txt" ]; then
        print_success "Both test outputs available for comparison"
        
        echo "Baseline output:"
        cat baseline_output.txt
        echo ""
        echo "Delay injector output:"
        cat delay_output.txt
        
        # Check if delay injector messages are present
        if grep -q "RCCL-DELAY" delay_output.txt; then
            print_success "Delay injector messages detected in output"
        else
            print_warning "No delay injector messages found in output"
        fi
    else
        print_error "Test output files not found"
        return 1
    fi
}

# Clean up test files
cleanup() {
    print_status "Cleaning up test files..."
    
    rm -f "$TEST_PROGRAM" "$TEST_SOURCE" baseline_output.txt delay_output.txt
    
    print_success "Cleanup completed"
}

# Main test function
run_tests() {
    print_status "Starting RCCL Delay Injector Tests"
    echo ""
    
    check_dependencies
    echo ""
    
    create_test_program
    echo ""
    
    compile_test_program
    echo ""
    
    run_baseline_test
    echo ""
    
    run_delay_test
    echo ""
    
    compare_results
    echo ""
    
    print_success "All tests completed successfully!"
}

# Show usage
show_usage() {
    echo "RCCL Delay Injector Test Script"
    echo ""
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  test     - Run all tests (default)"
    echo "  clean    - Clean up test files"
    echo "  help     - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 test   # Run all tests"
    echo "  $0 clean  # Clean up files"
    echo ""
}

# Main execution
main() {
    case "${1:-test}" in
        test)
            run_tests
            ;;
        clean)
            cleanup
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
