#!/bin/bash

# Enhanced Multi-Model Benchmark Script for Pi5 YOLO Object Detection
# Compares performance across all available models and formats

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 Pi5 Multi-Model YOLO Benchmark Suite"
echo "======================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Download models if needed
if [ ! -f "models/yolov8n.pt" ]; then
    echo "📥 Downloading models..."
    python scripts/download_models.py
fi

# Create results directory
RESULTS_DIR="benchmark_results"
mkdir -p "$RESULTS_DIR"

# Get system info
echo "📊 System Information:"
echo "---------------------"
echo "📌 OS: $(uname -a)"
echo "📌 Python: $(python --version)"
echo "📌 CPU: $(nproc) cores"
echo "📌 Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "📌 Temperature: $(vcgencmd measure_temp 2>/dev/null || echo 'N/A')"
echo "📌 CPU Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'N/A')"
echo ""

# List available models
echo "📦 Available Models:"
echo "-------------------"
find models/ -name "*.pt" -o -name "*.onnx" | sort | while read model; do
    size=$(du -h "$model" | cut -f1)
    echo "📌 $model ($size)"
done
echo ""

# Run comprehensive benchmark
echo "🔥 Running Multi-Model Benchmark..."
echo "-----------------------------------"

python -c "
import time
import psutil
import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
import json
from pathlib import Path
import glob
import traceback

# Configuration
test_duration = 20  # seconds per model
image_size = (640, 480)
results_dir = 'benchmark_results'
warmup_iterations = 5

# Get all available models
model_patterns = [
    'models/*.pt',
    'models/*.onnx'
]

all_models = []
for pattern in model_patterns:
    all_models.extend(glob.glob(pattern))

all_models = sorted(all_models)

if not all_models:
    print('❌ No models found!')
    sys.exit(1)

print(f'🎯 Found {len(all_models)} models to benchmark')
print(f'⏱️  Test duration per model: {test_duration} seconds')
print(f'🔥 Warmup iterations: {warmup_iterations}')
print('')

# Create dummy image (simulating camera input)
dummy_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)

overall_results = []
failed_models = []

for i, model_path in enumerate(all_models, 1):
    model_name = Path(model_path).name
    print(f'📦 [{i}/{len(all_models)}] Testing: {model_name}')
    
    try:
        # Load model
        start_load = time.time()
        model = YOLO(model_path, verbose=False)
        load_time = time.time() - start_load
        
        print(f'   📋 Model loaded in {load_time:.2f}s')
        
        # Warm up
        print(f'   🔥 Warming up ({warmup_iterations} iterations)...')
        warmup_times = []
        for j in range(warmup_iterations):
            start_warmup = time.time()
            _ = model(dummy_image, verbose=False)
            warmup_times.append(time.time() - start_warmup)
        
        avg_warmup_time = sum(warmup_times) / len(warmup_times)
        print(f'   🕒 Average warmup inference time: {avg_warmup_time*1000:.1f}ms')
        
        print(f'   ⚡ Running benchmark for {test_duration}s...')
        
        # Benchmark
        start_time = time.time()
        frame_count = 0
        cpu_usage = []
        memory_usage = []
        inference_times = []
        
        process = psutil.Process()
        
        while time.time() - start_time < test_duration:
            # Run inference with timing
            inference_start = time.time()
            results = model(dummy_image, verbose=False)
            inference_time = time.time() - inference_start
            
            frame_count += 1
            inference_times.append(inference_time)
            
            # Monitor system resources every 10 frames to reduce overhead
            if frame_count % 10 == 0:
                cpu_usage.append(psutil.cpu_percent())
                memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        avg_cpu = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
        avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0
        max_memory = max(memory_usage) if memory_usage else 0
        min_memory = min(memory_usage) if memory_usage else 0
        
        # Inference time statistics
        avg_inference_time = sum(inference_times) / len(inference_times)
        min_inference_time = min(inference_times)
        max_inference_time = max(inference_times)
        median_inference_time = sorted(inference_times)[len(inference_times)//2]
        
        # Get model info
        model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        
        # Detect model type and format
        model_format = 'PyTorch' if model_path.endswith('.pt') else 'ONNX'
        model_family = 'Unknown'
        if 'yolov8' in model_name:
            model_family = 'YOLOv8'
        elif 'yolov5' in model_name:
            model_family = 'YOLOv5'
        elif 'yolov10' in model_name:
            model_family = 'YOLOv10'
        
        model_size_category = 'Unknown'
        if 'n' in model_name.lower():
            model_size_category = 'Nano'
        elif 's' in model_name.lower():
            model_size_category = 'Small'
        elif 'm' in model_name.lower():
            model_size_category = 'Medium'
        elif 'l' in model_name.lower():
            model_size_category = 'Large'
        elif 'x' in model_name.lower():
            model_size_category = 'Extra Large'
        
        result = {
            'model_name': model_name,
            'model_path': model_path,
            'model_format': model_format,
            'model_family': model_family,
            'model_size_category': model_size_category,
            'model_size_mb': round(model_size_mb, 1),
            'load_time_s': round(load_time, 2),
            'test_duration': test_duration,
            'total_frames': frame_count,
            'total_time': round(total_time, 2),
            'average_fps': round(avg_fps, 2),
            'average_inference_time_ms': round(avg_inference_time * 1000, 1),
            'min_inference_time_ms': round(min_inference_time * 1000, 1),
            'max_inference_time_ms': round(max_inference_time * 1000, 1),
            'median_inference_time_ms': round(median_inference_time * 1000, 1),
            'average_cpu_usage': round(avg_cpu, 1),
            'average_memory_usage': round(avg_memory, 1),
            'peak_memory_usage': round(max_memory, 1),
            'min_memory_usage': round(min_memory, 1),
            'memory_variation': round(max_memory - min_memory, 1),
            'fps_per_mb': round(avg_fps / model_size_mb, 2),
            'fps_per_param': 'N/A',  # Would need model analysis
            'image_size': image_size,
            'status': 'success'
        }
        
        overall_results.append(result)
        
        print(f'   ✅ FPS: {avg_fps:.1f} | Inference: {avg_inference_time*1000:.1f}ms | CPU: {avg_cpu:.1f}% | Memory: {avg_memory:.1f}MB')
        
    except Exception as e:
        print(f'   ❌ Failed: {str(e)}')
        failed_result = {
            'model_name': model_name,
            'model_path': model_path,
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        failed_models.append(failed_result)
    
    print('')

# Display comprehensive summary
print('📈 COMPREHENSIVE BENCHMARK RESULTS')
print('==================================')
print('')

if overall_results:
    # Sort by FPS (descending)
    overall_results.sort(key=lambda x: x['average_fps'], reverse=True)
    
    print('🏆 Performance Ranking (by FPS):')
    print('--------------------------------')
    print(f'{"Rank":<4} {"Model":<25} {"FPS":<8} {"Inference":<12} {"Size":<8} {"CPU":<6} {"Memory":<8} {"Format":<8}')
    print('-' * 80)
    for i, result in enumerate(overall_results, 1):
        print(f'{i:<4} {result[\"model_name\"]:<25} '
              f'{result[\"average_fps\"]:>7.1f} '
              f'{result[\"average_inference_time_ms\"]:>8.1f}ms '
              f'{result[\"model_size_mb\"]:>6.1f}MB '
              f'{result[\"average_cpu_usage\"]:>5.1f}% '
              f'{result[\"average_memory_usage\"]:>6.1f}MB '
              f'{result[\"model_format\"]:>8}')
    
    print('')
    print('📊 Performance Categories:')
    print('-------------------------')
    
    # Group by performance categories
    real_time_models = [r for r in overall_results if r['average_fps'] >= 10]
    interactive_models = [r for r in overall_results if 5 <= r['average_fps'] < 10]
    batch_models = [r for r in overall_results if r['average_fps'] < 5]
    
    print(f'🚀 Real-time (≥10 FPS): {len(real_time_models)} models')
    for model in real_time_models[:3]:  # Top 3
        print(f'   • {model[\"model_name\"]}: {model[\"average_fps\"]:.1f} FPS')
    
    print(f'⚡ Interactive (5-10 FPS): {len(interactive_models)} models')
    for model in interactive_models[:3]:  # Top 3
        print(f'   • {model[\"model_name\"]}: {model[\"average_fps\"]:.1f} FPS')
    
    print(f'🎯 Batch processing (<5 FPS): {len(batch_models)} models')
    for model in batch_models[:3]:  # Top 3
        print(f'   • {model[\"model_name\"]}: {model[\"average_fps\"]:.1f} FPS')
    
    print('')
    print('🥇 Category Winners:')
    print('-------------------')
    
    # Best performers by category
    best_fps = max(overall_results, key=lambda x: x['average_fps'])
    smallest_model = min(overall_results, key=lambda x: x['model_size_mb'])
    lowest_cpu = min(overall_results, key=lambda x: x['average_cpu_usage'])
    lowest_memory = min(overall_results, key=lambda x: x['average_memory_usage'])
    fastest_inference = min(overall_results, key=lambda x: x['average_inference_time_ms'])
    best_efficiency = max(overall_results, key=lambda x: x['fps_per_mb'])
    
    print(f'🚀 Fastest FPS:       {best_fps[\"model_name\"]} ({best_fps[\"average_fps\"]:.1f} FPS)')
    print(f'📦 Smallest Model:    {smallest_model[\"model_name\"]} ({smallest_model[\"model_size_mb\"]:.1f} MB)')
    print(f'⚡ Lowest CPU:        {lowest_cpu[\"model_name\"]} ({lowest_cpu[\"average_cpu_usage\"]:.1f}%)')
    print(f'💾 Lowest Memory:     {lowest_memory[\"model_name\"]} ({lowest_memory[\"average_memory_usage\"]:.1f} MB)')
    print(f'⏱️ Fastest Inference: {fastest_inference[\"model_name\"]} ({fastest_inference[\"average_inference_time_ms\"]:.1f}ms)')
    print(f'⚖️ Best Efficiency:   {best_efficiency[\"model_name\"]} ({best_efficiency[\"fps_per_mb\"]:.2f} FPS/MB)')
    
    # Format comparison
    print('')
    print('📋 Format Comparison:')
    print('---------------------')
    
    pytorch_models = [r for r in overall_results if r['model_format'] == 'PyTorch']
    onnx_models = [r for r in overall_results if r['model_format'] == 'ONNX']
    
    if pytorch_models and onnx_models:
        avg_pytorch_fps = sum(r['average_fps'] for r in pytorch_models) / len(pytorch_models)
        avg_onnx_fps = sum(r['average_fps'] for r in onnx_models) / len(onnx_models)
        
        print(f'🔥 PyTorch (.pt):  {len(pytorch_models)} models, avg {avg_pytorch_fps:.1f} FPS')
        print(f'⚡ ONNX (.onnx):   {len(onnx_models)} models, avg {avg_onnx_fps:.1f} FPS')
        
        if avg_onnx_fps > avg_pytorch_fps:
            improvement = ((avg_onnx_fps - avg_pytorch_fps) / avg_pytorch_fps) * 100
            print(f'📈 ONNX provides {improvement:.1f}% average performance improvement')
        else:
            print('📊 No significant format-based performance difference')
    
    # Family comparison
    print('')
    print('👨‍👩‍👧‍👦 Model Family Comparison:')
    print('---------------------------')
    
    families = {}
    for result in overall_results:
        family = result['model_family']
        if family not in families:
            families[family] = []
        families[family].append(result)
    
    for family, models in families.items():
        if len(models) > 1:
            avg_fps = sum(m['average_fps'] for m in models) / len(models)
            best_model = max(models, key=lambda x: x['average_fps'])
            print(f'{family}: {len(models)} models, avg {avg_fps:.1f} FPS, best: {best_model[\"model_name\"]} ({best_model[\"average_fps\"]:.1f} FPS)')

if failed_models:
    print('')
    print('❌ Failed Models:')
    print('----------------')
    for failed in failed_models:
        print(f'   {failed[\"model_name\"]}: {failed[\"error\"]}')

# Save detailed results
timestamp = time.strftime('%Y%m%d_%H%M%S')
results_file = f'{results_dir}/multi_model_benchmark_{timestamp}.json'

final_results = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'test_configuration': {
        'test_duration_per_model': test_duration,
        'warmup_iterations': warmup_iterations,
        'image_size': image_size,
        'total_models_tested': len(all_models),
        'successful_tests': len(overall_results),
        'failed_tests': len(failed_models)
    },
    'system_info': {
        'cpu_cores': os.cpu_count(),
        'python_version': sys.version
    },
    'successful_results': overall_results,
    'failed_results': failed_models
}

with open(results_file, 'w') as f:
    json.dump(final_results, f, indent=2)

# Create CSV for easy analysis
csv_file = f'{results_dir}/benchmark_comparison_{timestamp}.csv'
with open(csv_file, 'w') as f:
    f.write('Model,Family,Format,Size_Category,Size_MB,FPS,Inference_ms,CPU_Percent,Memory_MB,FPS_per_MB\\n')
    for result in overall_results:
        f.write(f'{result[\"model_name\"]},{result[\"model_family\"]},{result[\"model_format\"]},'
                f'{result[\"model_size_category\"]},{result[\"model_size_mb\"]},{result[\"average_fps\"]},'
                f'{result[\"average_inference_time_ms\"]},{result[\"average_cpu_usage\"]},'
                f'{result[\"average_memory_usage\"]},{result[\"fps_per_mb\"]}\\n')

print(f'')
print(f'💾 Detailed results saved to: {results_file}')
print(f'📊 CSV analysis file: {csv_file}')
print(f'📁 Results directory: {results_dir}')
"

echo ""
echo "✅ Multi-model benchmark complete!"
echo ""
echo "🎯 Quick Recommendations:"
echo "========================"

# Extract top performing models
TOP_MODEL=$(ls benchmark_results/benchmark_comparison_*.csv 2>/dev/null | head -1)
if [ -f "$TOP_MODEL" ]; then
    echo "Based on benchmark results:"
    
    # Get fastest nano model for real-time use
    NANO_BEST=$(tail -n +2 "$TOP_MODEL" | grep -i nano | sort -t',' -k6 -nr | head -1 | cut -d',' -f1)
    if [ -n "$NANO_BEST" ]; then
        NANO_FPS=$(tail -n +2 "$TOP_MODEL" | grep "$NANO_BEST" | cut -d',' -f6)
        echo "🚀 Best real-time model: $NANO_BEST (${NANO_FPS} FPS)"
    fi
    
    # Get best efficiency model
    EFFICIENCY_BEST=$(tail -n +2 "$TOP_MODEL" | sort -t',' -k10 -nr | head -1 | cut -d',' -f1)
    if [ -n "$EFFICIENCY_BEST" ]; then
        EFFICIENCY_RATIO=$(tail -n +2 "$TOP_MODEL" | grep "$EFFICIENCY_BEST" | cut -d',' -f10)
        echo "⚖️ Most efficient model: $EFFICIENCY_BEST (${EFFICIENCY_RATIO} FPS/MB)"
    fi
    
    # Compare PyTorch vs ONNX
    PT_AVG=$(tail -n +2 "$TOP_MODEL" | grep ",PyTorch," | cut -d',' -f6 | awk '{sum+=$1; count++} END{if(count>0) print sum/count}')
    ONNX_AVG=$(tail -n +2 "$TOP_MODEL" | grep ",ONNX," | cut -d',' -f6 | awk '{sum+=$1; count++} END{if(count>0) print sum/count}')
    
    if [ -n "$PT_AVG" ] && [ -n "$ONNX_AVG" ]; then
        IMPROVEMENT=$(echo "scale=1; ($ONNX_AVG - $PT_AVG) / $PT_AVG * 100" | bc -l 2>/dev/null || echo "0")
        echo "📈 ONNX vs PyTorch improvement: ${IMPROVEMENT}%"
    fi
fi

echo ""
echo "🔧 Optimization Next Steps:"
echo "   • Use nano models for real-time applications (>10 FPS)"
echo "   • ONNX models typically provide better performance"
echo "   • Consider quantization for additional 2-4x speedup"
echo "   • Monitor temperature during extended operation"
echo "   • Adjust image resolution based on FPS requirements"
