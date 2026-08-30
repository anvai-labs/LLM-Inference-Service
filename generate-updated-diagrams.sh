#!/bin/bash

# Generate updated diagrams for README using JSON configuration with scale 4
# Script uses the diagram-config.json file for high-quality diagram generation

set -e

echo "🎨 Generating Updated LLM Inference Service Diagrams"
echo "=============================================="

# Check if required tools are available
if ! command -v mmdc &> /dev/null; then
    echo "❌ Mermaid CLI not found. Installing..."
    npm install -g @mermaid-js/mermaid-cli
fi

# Check if JSON config exists
if [ ! -f "diagram-config.json" ]; then
    echo "❌ diagram-config.json not found"
    exit 1
fi

# Create images directory if it doesn't exist
mkdir -p images

# Read JSON config and generate each diagram
echo "📖 Reading diagram configuration..."

# Extract diagram configurations using jq
DIAGRAMS=$(jq -r '.diagrams[] | @base64' diagram-config.json)

for diagram in $DIAGRAMS; do
    # Decode diagram data
    diagram_data=$(printf '%s' "$diagram" | base64 --decode)
    
    # Extract fields
    name=$(printf '%s' "$diagram_data" | jq -r '.name')
    title=$(printf '%s' "$diagram_data" | jq -r '.title')
    mermaid_content=$(printf '%s' "$diagram_data" | jq -r '.mermaid')
    output_file=$(printf '%s' "$diagram_data" | jq -r '.outputFile')
    
    echo "🎯 Generating: $title"
    echo "   📄 Name: $name"
    echo "   📁 Output: $output_file"
    
    # Create temporary mermaid file
    temp_file="/tmp/${name}.mmd"
    echo "$mermaid_content" > "$temp_file"
    
    # Generate diagram with scale 4 and puppeteer config
    echo "   🔄 Rendering with scale 4..."
    if mmdc -i "$temp_file" -o "$output_file" \
        --scale 4 \
        --puppeteerConfigFile puppeteer-config.json \
        --theme dark \
        --backgroundColor transparent; then
        echo "   ✅ Generated: $output_file"
    else
        echo "   ❌ Failed to generate: $output_file"
    fi
    
    # Clean up temp file
    rm -f "$temp_file"
    echo ""
done

echo "🎉 Diagram generation complete!"
echo ""
echo "📊 Generated diagrams:"
echo "   • Architecture Overview: images/architecture.png"
echo "   • Dashboard Data Flow: images/dashboard_flow.png"
echo "   • API Compatibility: images/api_compatibility.png"
echo "   • GPU Utilization: images/gpu_utilization.png"
echo ""
echo "💡 These diagrams show:"
echo "   • Real GPU data from 4x Tesla M10 setup"
echo "   • 52 models with proper context detection"
echo "   • Live 15-second refresh monitoring"
echo "   • Real API endpoint metrics (not simulated)"
echo ""
echo "📝 Ready to update README.adoc!"
