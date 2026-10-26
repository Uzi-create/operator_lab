CXX ?= c++
CXXFLAGS ?= -O2
CPPFLAGS ?=
LDFLAGS ?=
BUILD_DIR ?= build
BASE_FLAGS = -std=c++17 -Wall -Wextra -Wpedantic -Iinclude
HEADERS = $(wildcard include/operator_lab/*.hpp)

.PHONY: all test demo clean
all: $(BUILD_DIR)/test_rgbd $(BUILD_DIR)/depth_target_demo

$(BUILD_DIR):
	mkdir -p "$@"

$(BUILD_DIR)/test_rgbd: tests/test_rgbd.cpp tests/header_link.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(BASE_FLAGS) $(CXXFLAGS) tests/test_rgbd.cpp tests/header_link.cpp $(LDFLAGS) -o "$@"

$(BUILD_DIR)/depth_target_demo: examples/depth_target_demo.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(BASE_FLAGS) $(CXXFLAGS) examples/depth_target_demo.cpp $(LDFLAGS) -o "$@"

test: $(BUILD_DIR)/test_rgbd
	"$(BUILD_DIR)/test_rgbd"

demo: $(BUILD_DIR)/depth_target_demo
	"$(BUILD_DIR)/depth_target_demo"

clean:
	rm -f "$(BUILD_DIR)/test_rgbd" "$(BUILD_DIR)/depth_target_demo"
