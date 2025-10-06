# RCCL Delay Injector Makefile

CXX ?= g++
CXXFLAGS ?= -O2 -fPIC -shared -std=c++17 -Wall -Wextra -Wpedantic
LDFLAGS ?= -ldl -lhip_hcc

TARGET = libdelay_injector.so
SOURCE = delay_injector.cpp

PREFIX ?= /usr/local
LIBDIR = $(PREFIX)/lib

.PHONY: all clean distclean install uninstall debug release test-compile check help

all: $(TARGET)
	@echo "Built $(TARGET)"

$(TARGET): $(SOURCE)
	@echo "Compiling $(SOURCE)..."
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)
	@echo "Compilation completed"

clean:
	@echo "Removing build artifacts..."
	rm -f $(TARGET)
	@echo "Clean completed"

distclean: clean
	@echo "Removing all generated files..."
	rm -f *.so *.o *.a
	@echo "Distclean completed"

install: $(TARGET)
	@echo "Installing $(TARGET) to $(LIBDIR)..."
	@mkdir -p $(LIBDIR)
	cp $(TARGET) $(LIBDIR)/
	@echo "Installation completed"
	@echo "Use: LD_PRELOAD=$(LIBDIR)/$(TARGET) your_program"

uninstall:
	@echo "Removing $(TARGET) from $(LIBDIR)..."
	rm -f $(LIBDIR)/$(TARGET)
	@echo "Uninstall completed"

debug: CXXFLAGS += -g -DDEBUG -O0
debug: $(TARGET)
	@echo "Debug build completed"

release: CXXFLAGS += -DNDEBUG -O3 -flto
release: $(TARGET)
	@echo "Release build completed"

test-compile: $(TARGET)
	@echo "Compilation test passed"

check: $(TARGET)
	@echo "Running basic checks..."
	@ldd $(TARGET) > /dev/null 2>&1 && echo "Library dependencies OK" || echo "Library dependencies missing"
	@nm -D $(TARGET) | grep -q "ncclAllReduce" && echo "RCCL symbols found" || echo "RCCL symbols missing"
	@echo "Basic checks completed"

help:
	@echo "RCCL Delay Injector Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build the delay injector library (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  distclean    - Remove all generated files"
	@echo "  install      - Install library to system"
	@echo "  uninstall    - Remove library from system"
	@echo "  debug        - Build debug version"
	@echo "  release      - Build optimized release version"
	@echo "  test-compile - Test compilation"
	@echo "  check        - Run basic checks"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make                    # Build the library"
	@echo "  make clean              # Clean up"
	@echo "  make install            # Install to system"
	@echo "  make debug              # Build debug version"
	@echo "  make release            # Build optimized version"
	@echo ""
	@echo "Environment Variables:"
	@echo "  CXX         - C++ compiler (default: g++)"
	@echo "  CXXFLAGS    - Compiler flags"
	@echo "  LDFLAGS     - Linker flags"
	@echo "  PREFIX      - Installation prefix (default: /usr/local)"
