#!/bin/bash
# Performance benchmark script for the detection pipeline

set -e

echo "=== Real-Time Object Detection Pipeline - Performance Benchmark ==="
echo "Date: $(date)"
echo ""

# Setup
cd "$(dirname "$0")/.."
source venv/bin/activate 2>/dev/null || echo "Warning: Virtual environment not found"
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Configuration
DURATION=30  # Test duration in seconds
OUTPUT_DIR="/tmp/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Benchmark duration: ${DURATION} seconds"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to run benchmark
run_benchmark() {
    local mode="$1"
    local cmd="$2"
    local desc="$3"
    
    echo "=== Benchmarking $desc ==="
    echo "Command: $cmd"
    echo "Starting benchmark..."
    
    # Run in background and capture output
    timeout ${DURATION}s $cmd > "$OUTPUT_DIR/${mode}.log" 2>&1 &
    local pid=$!
    
    # Monitor performance
    echo "Monitoring performance for ${DURATION} seconds..."
    local start_time=$(date +%s)
    local samples=0
    local total_cpu=0
    local total_mem=0
    local max_temp=0
    
    while kill -0 $pid 2>/dev/null; do
        local cpu_usage=$(ps -p $pid -o %cpu --no-headers 2>/dev/null || echo "0")
        local mem_usage=$(ps -p $pid -o %mem --no-headers 2>/dev/null || echo "0")
        local temp=$(vcgencmd measure_temp | grep -o '[0-9.]*')
        
        total_cpu=$(echo "$total_cpu + $cpu_usage" | bc -l 2>/dev/null || echo "$total_cpu")
        total_mem=$(echo "$total_mem + $mem_usage" | bc -l 2>/dev/null || echo "$total_mem")
        
        if (( $(echo "$temp > $max_temp" | bc -l) )); then
            max_temp=$temp
        fi
        
        samples=$((samples + 1))
        sleep 1
        
        # Show progress
        local elapsed=$(($(date +%s) - start_time))
        printf "\rProgress: %d/%d seconds, CPU: %.1f%%, Mem: %.1f%%, Temp: %.1f°C" \
               $elapsed $DURATION $cpu_usage $mem_usage $temp
    done
    
    echo ""
    
    # Calculate averages
    local avg_cpu=$(echo "scale=2; $total_cpu / $samples" | bc -l 2>/dev/null || echo "0")
    local avg_mem=$(echo "scale=2; $total_mem / $samples" | bc -l 2>/dev/null || echo "0")
    
    # Extract FPS from log file
    local fps="N/A"
    if [ -f "$OUTPUT_DIR/${mode}.log" ]; then
        fps=$(grep -o "FPS: [0-9.]*" "$OUTPUT_DIR/${mode}.log" | tail -1 | cut -d' ' -f2 || echo "N/A")
    fi
    
    # Save results
    cat >> "$OUTPUT_DIR/benchmark_results.txt" << EOF

=== $desc ===
Duration: ${DURATION} seconds
Average CPU: ${avg_cpu}%
Average Memory: ${avg_mem}%
Max Temperature: ${max_temp}°C
Estimated FPS: $fps
Command: $cmd

EOF
    
    echo "Results:"
    echo "  Average CPU usage: ${avg_cpu}%"
    echo "  Average memory usage: ${avg_mem}%"
    echo "  Max temperature: ${max_temp}°C"
    echo "  Estimated FPS: $fps"
    echo ""
}

# System baseline
echo "=== System Information ==="
echo "CPU: $(nproc) cores"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Temperature: $(vcgencmd measure_temp)"
echo "CPU Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
echo ""

# Benchmark different modes
echo "Starting performance benchmarks..."
echo ""

# 1. Standard mode
run_benchmark "standard" \
              "python3 src/main.py --headless --width 424 --height 240" \
              "Standard Mode (424x240)"

# 2. High-performance mode
if [ -f "src/high_performance_viewer.py" ]; then
    run_benchmark "high_perf" \
                  "python3 src/high_performance_viewer.py --headless" \
                  "High-Performance Mode"
fi

# 3. Web viewer mode
if [ -f "src/web_viewer.py" ]; then
    echo "=== Web Viewer Benchmark ==="
    echo "Starting web viewer in background..."
    python3 src/web_viewer.py --width 424 --height 240 > "$OUTPUT_DIR/web_viewer.log" 2>&1 &
    local web_pid=$!
    sleep 5  # Let it start up
    
    # Test with curl
    echo "Testing web interface..."
    for i in {1..10}; do
        curl -s http://localhost:5000 > /dev/null && echo "Web request $i: OK" || echo "Web request $i: FAILED"
        sleep 1
    done
    
    kill $web_pid 2>/dev/null || true
    wait $web_pid 2>/dev/null || true
    echo ""
fi

# 4. Mock camera test (for comparison)
run_benchmark "mock" \
              "python3 -c \"
import sys; sys.path.append('src')
from mock_camera import MockRealSenseStreamer
from detector import Detector
import time

camera = MockRealSenseStreamer(424, 240, 30)
detector = Detector()
detector.load_model()

camera.start()
fps_counter = 0
start_time = time.time()

try:
    while time.time() - start_time < 30:
        frame_data = camera.get_frame()
        if frame_data:
            frame, _ = frame_data
            detections = detector.detect(frame)
            fps_counter += 1
            
            if fps_counter % 30 == 0:
                elapsed = time.time() - start_time
                fps = fps_counter / elapsed
                print(f'FPS: {fps:.2f}')
finally:
    camera.stop()
\"" \
              "Mock Camera Test (Baseline)"

# 5. Run pytest benchmarks if available
if command -v pytest >/dev/null 2>&1; then
    echo "=== Running pytest benchmarks ==="
    pytest tests/test_fps.py -v --benchmark-only > "$OUTPUT_DIR/pytest_benchmark.log" 2>&1 || true
fi

# Generate summary report
echo "=== Benchmark Summary ==="
cat "$OUTPUT_DIR/benchmark_results.txt"

# Create CSV for analysis
echo "Mode,FPS,CPU_Avg,Memory_Avg,Max_Temp" > "$OUTPUT_DIR/benchmark_summary.csv"
grep -A 10 "===" "$OUTPUT_DIR/benchmark_results.txt" | grep -E "(FPS|CPU|Memory|Temperature)" | \
    awk 'BEGIN{mode=""} /===/{getline; mode=$0} /FPS/{fps=$NF} /CPU/{cpu=$NF} /Memory/{mem=$NF} /Temperature/{temp=$NF; print mode","fps","cpu","mem","temp}' \
    >> "$OUTPUT_DIR/benchmark_summary.csv" 2>/dev/null || true

# System health after benchmark
echo ""
echo "=== Post-Benchmark System Health ==="
echo "Temperature: $(vcgencmd measure_temp)"
echo "Throttling: $(vcgencmd get_throttled)"
echo "Memory usage:"
free -h

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: $OUTPUT_DIR/"
echo "Summary: $OUTPUT_DIR/benchmark_results.txt"
echo "CSV data: $OUTPUT_DIR/benchmark_summary.csv"
echo ""
echo "Recommendations:"

# Basic recommendations based on results
if [ -f "$OUTPUT_DIR/benchmark_results.txt" ]; then
    if grep -q "Temperature: [8-9][0-9]" "$OUTPUT_DIR/benchmark_results.txt"; then
        echo "⚠ High temperature detected - consider improving cooling"
    fi
    
    local best_fps=$(grep "FPS:" "$OUTPUT_DIR/benchmark_results.txt" | grep -v "N/A" | sort -nr -k2 | head -1)
    if [ -n "$best_fps" ]; then
        echo "✓ Best performance mode based on FPS"
    fi
fi

echo ""
echo "For detailed analysis, examine log files in $OUTPUT_DIR/"
echo "For optimization tips, see docs/performance_optimization.md"
