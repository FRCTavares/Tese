#!/bin/bash

# Project cleanup script - removes temporary files and caches
# Run this periodically to keep the project clean

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🧹 Cleaning up temporary files and caches..."
echo "============================================="

# Remove Python cache files
echo "📂 Removing Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -type f -delete 2>/dev/null || true
find . -name "*.pyo" -type f -delete 2>/dev/null || true

# Remove test cache
echo "🧪 Removing test cache..."
rm -rf .pytest_cache/ .tox/ .coverage .coverage.* htmlcov/ 2>/dev/null || true

# Remove build artifacts
echo "🔨 Removing build artifacts..."
rm -rf build/ dist/ *.egg-info/ .eggs/ 2>/dev/null || true

# Remove temporary files
echo "📄 Removing temporary files..."
find . -name "*.tmp" -type f -delete 2>/dev/null || true
find . -name "*.log" -type f -delete 2>/dev/null || true
find . -name "*.bak" -type f -delete 2>/dev/null || true
find . -name "*~" -type f -delete 2>/dev/null || true

# Remove empty directories (except important ones)
echo "📁 Removing empty directories..."
find . -type d -empty -not -path "./venv/*" -not -path "./models" -not -path "./config" -not -path "./docs" -not -path "./src" -not -path "./scripts" -not -path "./tests" -not -path "./notebooks" -not -path "./requirements" -delete 2>/dev/null || true

# Clean up old benchmark results (keep last 5)
if [ -d "benchmark_results" ]; then
    echo "📊 Cleaning old benchmark results (keeping last 5)..."
    cd benchmark_results
    ls -t *.json 2>/dev/null | tail -n +6 | xargs -r rm -f
    ls -t *.csv 2>/dev/null | tail -n +6 | xargs -r rm -f
    cd ..
fi

# Clean up pip cache (if in virtual environment)
if [ -n "$VIRTUAL_ENV" ]; then
    echo "📦 Cleaning pip cache..."
    pip cache purge >/dev/null 2>&1 || true
fi

# Summary
echo ""
echo "✅ Cleanup complete!"
echo ""
echo "🎯 Project structure maintained:"
echo "   • Python cache files removed"
echo "   • Test artifacts cleaned"
echo "   • Temporary files deleted"
echo "   • Empty directories removed"
echo "   • Old benchmark results archived"
echo ""
echo "💡 To keep the project clean, run this script periodically:"
echo "   ./scripts/cleanup.sh"
