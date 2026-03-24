#!/bin/bash
# Run both API and Streamlit frontend

echo "Starting Mushroom Classifier..."
echo "================================"

# Start API in background
echo "Starting API server on http://localhost:8000..."
python run_api.py &
API_PID=$!

# Wait for API to start
sleep 3

# Start Streamlit
echo "Starting Streamlit on http://localhost:8501..."
python run_app.py &
STREAMLIT_PID=$!

echo ""
echo "================================"
echo "Services running:"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Frontend: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop all services"
echo "================================"

# Handle Ctrl+C
trap "echo 'Stopping services...'; kill $API_PID $STREAMLIT_PID 2>/dev/null; exit 0" INT

# Wait for both processes
wait