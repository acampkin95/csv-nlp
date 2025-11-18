#!/bin/bash
# Simple startup script for Message Processor

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  MESSAGE PROCESSOR - ANALYSIS SYSTEM${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
    exit 1
fi

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Usage: ./analyze.sh <csv_file> [options]${NC}"
    echo ""
    echo "Options:"
    echo "  -o <dir>         Output directory (default: Reports)"
    echo "  -c <preset>      Config preset: quick_analysis, deep_analysis, clinical_report, legal_report"
    echo "  --use-sqlite     Use local SQLite instead of PostgreSQL"
    echo "  --no-grooming    Disable grooming detection"
    echo "  --no-manipulation Disable manipulation detection"
    echo "  --no-deception   Disable deception analysis"
    echo "  -v              Verbose output"
    echo ""
    echo "Examples:"
    echo "  ./analyze.sh chat.csv"
    echo "  ./analyze.sh chat.csv -c quick_analysis"
    echo "  ./analyze.sh chat.csv -o MyReports --use-sqlite"
    exit 1
fi

# Run the processor
python3 message_processor.py "$@"