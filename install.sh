#!/bin/bash
# Installation script for Message Processor

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Message Processor Installation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
if command -v python3 &> /dev/null; then
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo -e "${GREEN}✓ Python $python_version found${NC}"
else
    echo -e "${RED}✗ Python 3 is required but not installed.${NC}"
    echo "Please install Python 3.8 or higher."
    exit 1
fi

# Install core dependencies
echo ""
echo -e "${YELLOW}Installing core dependencies...${NC}"
echo "This will install: psycopg2-binary, pandas, chardet"
echo -e "${YELLOW}Continue? (y/n)${NC}"
read -r response

if [[ "$response" == "y" ]]; then
    pip3 install --user psycopg2-binary pandas chardet 2>&1 | tail -5

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Core dependencies installed${NC}"
    else
        echo -e "${YELLOW}Note: You may need to use 'pip3 install --break-system-packages' on some systems${NC}"
    fi
else
    echo "Skipping core dependencies"
fi

# Install NLP dependencies (optional)
echo ""
echo -e "${YELLOW}Install NLP analysis modules? (Recommended)${NC}"
echo "This will install: vaderSentiment, textblob, nrclex, nltk"
echo -e "${YELLOW}Continue? (y/n)${NC}"
read -r response

if [[ "$response" == "y" ]]; then
    echo "Installing NLP modules..."
    pip3 install --user vaderSentiment textblob nrclex nltk 2>&1 | tail -5

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ NLP modules installed${NC}"

        # Download NLTK data
        echo "Downloading NLTK data..."
        python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('vader_lexicon', quiet=True)" 2>/dev/null
        echo -e "${GREEN}✓ NLTK data downloaded${NC}"
    fi
else
    echo "Skipping NLP modules (some features will be unavailable)"
fi

# Test PostgreSQL connection
echo ""
echo -e "${YELLOW}Testing PostgreSQL connection...${NC}"
python3 -c "
import psycopg2
try:
    conn = psycopg2.connect(
        host='acdev.host',
        database='messagestore',
        user='msgprocess',
        password='DHifde93jes9dk'
    )
    print('✓ PostgreSQL connection successful')
    conn.close()
except Exception as e:
    print('✗ PostgreSQL connection failed:', str(e)[:50])
    print('  You can still use --use-sqlite for local processing')
" 2>/dev/null

# Create necessary directories
echo ""
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p Reports logs data
echo -e "${GREEN}✓ Directories created${NC}"

# Make scripts executable
chmod +x analyze.sh 2>/dev/null

# Test import
echo ""
echo -e "${YELLOW}Testing system modules...${NC}"
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, 'src')
try:
    from validation.csv_validator import CSVValidator
    from config.config_manager import ConfigManager
    print('✓ Core modules working')
except ImportError as e:
    print('✗ Module import failed:', e)
" 2>/dev/null

# Final summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To analyze a CSV file, run:"
echo -e "${BLUE}  ./analyze.sh your_messages.csv${NC}"
echo ""
echo "For help:"
echo -e "${BLUE}  ./analyze.sh${NC}"
echo ""
echo "Example with options:"
echo -e "${BLUE}  ./analyze.sh messages.csv -c quick_analysis -o Reports/${NC}"
echo ""
echo -e "${YELLOW}Note: If you encounter permission errors with pip, you may need to:${NC}"
echo "  • Use a virtual environment (recommended)"
echo "  • Or use pip3 install --break-system-packages (not recommended)"
echo ""