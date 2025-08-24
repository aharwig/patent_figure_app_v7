#!/bin/bash
# run_patent_appv7.sh
# Launch Patent Figure Converter v7 with dependency check

echo "Activating virtual environment..."
source ./venv/bin/activate

echo "Checking required Python packages..."
REQUIRED_PACKAGES=("streamlit" "numpy" "pillow" "pdf2image" "pytesseract" "opencv-python" "scipy" "reportlab")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! python -c "import $pkg" &> /dev/null; then
        echo "Installing missing package: $pkg"
        pip install "$pkg"
    fi
done

echo ""
echo "Starting Patent Figure Converter v7..."
echo ""

# Auto-open Streamlit app in default browser
export BROWSER="open"   # macOS. Use "xdg-open" for Linux
export STREAMLIT_BROWSER_GATHER=false

# Launch the app
streamlit run patent_figure_app_v7.py --server.port 8501 --server.address localhost

# Deactivate venv when Streamlit stops
deactivate
