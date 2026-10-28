CXX ?= c++
CXXFLAGS ?= -O2
CPPFLAGS ?=
LDFLAGS ?=
BUILD_DIR ?= build
BASE_FLAGS = -std=c++17 -Wall -Wextra -Wpedantic -Iinclude
HEADERS = $(wildcard include/operator_lab/*.hpp)

.PHONY: all test demo replay navigation benchmark clean
all: $(BUILD_DIR)/test_performance $(BUILD_DIR)/test_rgbd $(BUILD_DIR)/test_tracking $(BUILD_DIR)/test_replay_io $(BUILD_DIR)/test_navigation $(BUILD_DIR)/depth_target_demo $(BUILD_DIR)/rgbd_replay $(BUILD_DIR)/navigation_replay

$(BUILD_DIR):
	mkdir -p "$@"

$(BUILD_DIR)/test_rgbd: tests/test_rgbd.cpp tests/header_link.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(BASE_FLAGS) $(CXXFLAGS) tests/test_rgbd.cpp tests/header_link.cpp $(LDFLAGS) -o "$@"

$(BUILD_DIR)/depth_target_demo: examples/depth_target_demo.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(BASE_FLAGS) $(CXXFLAGS) examples/depth_target_demo.cpp $(LDFLAGS) -o "$@"

$(BUILD_DIR)/test_tracking: tests/test_tracking.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(BASE_FLAGS) $(CXXFLAGS) tests/test_tracking.cpp $(LDFLAGS) -o "$@"

$(BUILD_DIR)/test_replay_io: tests/test_replay_io.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(BASE_FLAGS) $(CXXFLAGS) tests/test_replay_io.cpp $(LDFLAGS) -o "$@"

$(BUILD_DIR)/rgbd_replay: examples/rgbd_replay.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(BASE_FLAGS) $(CXXFLAGS) examples/rgbd_replay.cpp $(LDFLAGS) -o "$@"

$(BUILD_DIR)/test_navigation: tests/test_navigation.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(BASE_FLAGS) $(CXXFLAGS) tests/test_navigation.cpp $(LDFLAGS) -o "$@"

$(BUILD_DIR)/navigation_replay: examples/navigation_replay.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(BASE_FLAGS) $(CXXFLAGS) examples/navigation_replay.cpp $(LDFLAGS) -o "$@"

$(BUILD_DIR)/test_performance: tests/test_performance.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(BASE_FLAGS) $(CXXFLAGS) tests/test_performance.cpp $(LDFLAGS) -o "$@"

$(BUILD_DIR)/operator_benchmark: benchmarks/benchmark.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(BASE_FLAGS) $(CXXFLAGS) -O3 -DNDEBUG benchmarks/benchmark.cpp $(LDFLAGS) -o "$@"

benchmark: $(BUILD_DIR)/operator_benchmark
	"$(BUILD_DIR)/operator_benchmark"

test: $(BUILD_DIR)/test_performance $(BUILD_DIR)/test_rgbd $(BUILD_DIR)/test_tracking $(BUILD_DIR)/test_replay_io $(BUILD_DIR)/test_navigation
	"$(BUILD_DIR)/test_rgbd"
	"$(BUILD_DIR)/test_tracking"
	"$(BUILD_DIR)/test_replay_io"
	"$(BUILD_DIR)/test_navigation"
	"$(BUILD_DIR)/test_performance"

demo: $(BUILD_DIR)/depth_target_demo
	"$(BUILD_DIR)/depth_target_demo"

replay: $(BUILD_DIR)/rgbd_replay

navigation: $(BUILD_DIR)/navigation_replay

clean:
	rm -f "$(BUILD_DIR)/test_performance" "$(BUILD_DIR)/operator_benchmark" "$(BUILD_DIR)/test_rgbd" "$(BUILD_DIR)/test_tracking" "$(BUILD_DIR)/test_replay_io" "$(BUILD_DIR)/test_navigation" "$(BUILD_DIR)/depth_target_demo" "$(BUILD_DIR)/rgbd_replay" "$(BUILD_DIR)/navigation_replay"
