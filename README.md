# ChartViz 📊

**Advanced Chart Question Answering with Explainable AI**

ChartViz is a modern, interactive web application for chart-based question answering with attention visualization and explainability features. It supports multiple state-of-the-art vision-language models and provides detailed insights into how AI models interpret and analyze charts.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

---

## ✨ Features

- 🤖 **Multiple AI Models**: Support for LLaVA, LLaVA-NeXT, Qwen2.5-VL, and UniChart models
- 🔍 **Attention Visualization**: Interactive heatmaps showing where models focus
- 📈 **Advanced Analytics**: Token importance analysis, attention statistics, and layer-wise analysis
- 🎨 **Modern UI**: Clean, responsive interface built with Dash and Bootstrap
- 💾 **Offline Mode**: Run completely offline with locally downloaded models
- 📊 **Dataset Support**: Built-in support for ChartQA, PlotQA, and custom datasets
- 🔬 **Explainability**: Cross-attention extraction and visualization for model interpretability

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with 12+ GB VRAM (recommended)
- **Disk Space**: 20-50 GB for models
- **RAM**: 16 GB minimum, 32 GB recommended

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/ChartViz.git
cd ChartViz
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download models** (choose one method):

**Option A - Interactive Menu**:
```bash
python download_models.py
```

**Option B - Direct Download**:
```bash
# For LLaVA-NeXT (recommended)
python download_models.py --model llava_next_mistral_7b

# For other models
python download_models.py --model unichart
```

4. **Run the application**:
```bash
python app.py
```

5. **Open in browser**:
Navigate to `http://localhost:7860`

---

## 📦 Supported Models

| Model | Size | Resolution | Best For | Download Command |
|-------|------|------------|----------|------------------|
| **LLaVA-NeXT-Mistral-7B** | 14 GB | 672×672 | General chart QA | `python download_models.py --model llava_next_mistral_7b` |
| LLaVA v1.5 | 13 GB | 336×336 | Fast inference | `python download_models.py --model llava_v1_5_7b` |
| Qwen2.5-VL | 8 GB | Variable | OCR & text-heavy charts | `python download_models.py --model qwen25_vl_7b` |
| UniChart | 1.5 GB | 960×960 | Specialized chart QA | `python download_models.py --model unichart` |

### Model Comparison

**LLaVA-NeXT** (Recommended):
- ✅ High resolution (672×672 pixels)
- ✅ Enhanced visual reasoning
- ✅ Better chart understanding
- ✅ Longer context (2048 tokens)
- ✅ Cross-attention extraction support

**UniChart**:
- ✅ Lightweight and fast
- ✅ Specialized for charts
- ✅ Good for simple QA tasks
- ⚠️ Lower resolution

**Qwen2.5-VL**:
- ✅ Excellent OCR capabilities
- ✅ Variable resolution support
- ✅ Good for text-heavy charts
- ⚠️ Larger memory footprint

---

## 🎯 Usage

### Basic Workflow

1. **Select a Model**: Choose from the dropdown in the sidebar
2. **Load Model**: Click "Load Model" button
3. **Upload Chart**: Upload a chart image (PNG, JPG, etc.)
4. **Ask Question**: Enter your question about the chart
5. **Generate Answer**: Click "Generate Answer" to get AI response
6. **Visualize Attention**: Click "Extract Attention" to see where the model focuses

### Example Questions

- "What is the value for 2020?"
- "Which category has the highest value?"
- "What is the trend over time?"
- "Compare the values between A and B"
- "What is the total sum of all values?"

### Advanced Features

**Attention Heatmap**:
- Overlay showing model's visual focus
- Helps understand what the model "sees"
- Useful for debugging incorrect predictions

**Token Importance Analysis**:
- Shows which words in your question matter most
- Helps optimize question phrasing
- Identifies potential tokenization issues

**Statistical Metrics**:
- Attention entropy (focus vs. diffusion)
- Gini coefficient (concentration)
- Sparsity measures
- Layer-wise analysis

---

## 📚 Model Setup Guides

### LLaVA-NeXT Setup

**Quick Download**:
```bash
python download_models.py --model llava_next_mistral_7b
```

**Interactive Download**:
```bash
python download_models.py
# Then select option 3 and choose llava_next_mistral_7b
```

**Manual Download** (alternative):
```bash
pip install huggingface-hub[cli]
huggingface-cli download llava-hf/llava-v1.6-mistral-7b-hf \
    --local-dir ./LocalModels/LLaVA_NeXT_Mistral_7B \
    --local-dir-use-symlinks False
```

**Verification**:
```bash
ls LocalModels/LLaVA_NeXT_Mistral_7B/
# Should see: config.json, preprocessor_config.json, model files
```

**System Requirements**:
- GPU: 12+ GB VRAM (RTX 3060 12GB or better)
- Disk: 20 GB free space
- Time: 10-30 minutes (depending on internet speed)

### Using in ChartViz

1. Start the app: `python app.py`
2. Select "LLaVA-NeXT-Mistral-7B" from dropdown
3. Click "Load Model"
4. Upload chart and ask questions!

---

## 🔧 Configuration

### Model Configuration (`config.py`)

```python
# Adjust these settings based on your hardware
DEVICE = "cuda"  # or "cpu" for CPU-only mode
TORCH_DTYPE = "float16"  # or "float32" for CPU
MAX_NEW_TOKENS = 512  # Maximum answer length
BATCH_SIZE = 1  # Increase for batch processing
```

### Performance Optimization

**For Speed**:
```python
max_new_tokens = 256  # Shorter answers
num_beams = 1  # Greedy decoding
use_cache = True  # Enable KV caching
```

**For Quality**:
```python
max_new_tokens = 512  # Longer answers
num_beams = 4  # Beam search
temperature = 0.7  # More diverse outputs
```

**For Memory**:
```python
torch_dtype = torch.float16  # Half precision
device_map = "auto"  # Automatic device placement
low_cpu_mem_usage = True  # Reduce CPU memory
```

---

## 🐛 Troubleshooting

### Common Issues

**"Out of memory" error**:
```bash
# Solution 1: Reduce generation length
max_new_tokens = 256

# Solution 2: Use CPU mode (slower)
device = "cpu"

# Solution 3: Close other applications
# Free up GPU memory
```

**"Model not found" error**:
```bash
# Verify model path
ls LocalModels/LLaVA_NeXT_Mistral_7B/

# Re-download if needed
python download_models.py --model llava_next_mistral_7b
```

**"Download failed" error**:
```bash
# Check internet connection
# Try using VPN if Hugging Face is blocked
# Use alternative download method:
python download_models.py --model llava_next_mistral_7b
```

**Slow inference**:
```bash
# Check GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# Verify CUDA drivers installed
nvidia-smi

# Reduce max_new_tokens for faster generation
```

**Import errors**:
```bash
# Update dependencies
pip install --upgrade transformers torch pillow

# Or reinstall
pip install -r requirements.txt --force-reinstall
```

---

## 📊 Dataset Support

ChartViz supports multiple chart QA datasets:

### ChartQA
- 9,608 human-written questions
- 23,111 machine-generated questions
- Various chart types (bar, line, pie, etc.)

### PlotQA
- 28 million questions
- 224,377 plots
- Synthetic data generation

### Custom Datasets
Upload your own chart images and ask questions!

### Downloading Datasets

```bash
# Interactive menu
python download_datasets.py

# Direct download
python download_datasets.py --dataset chartqa
```

---

## 🏗️ Project Structure

```
ChartViz/
├── app.py                          # Main application entry point
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── download_models.py              # Model download manager (all models)
├── download_datasets.py            # Dataset downloader
├── models/                         # Model implementations
│   ├── base_model.py              # Base model interface
│   ├── llava_model.py             # LLaVA implementation
│   ├── llava_next_model.py        # LLaVA-NeXT implementation
│   ├── qwen25_vl_model.py         # Qwen2.5-VL implementation
│   ├── unichart_model.py          # UniChart implementation
│   └── model_manager.py           # Model loading/management
├── components/                     # UI components
│   ├── input_section.py           # Input controls
│   ├── results_section.py         # Results display
│   ├── attention_visualizer.py    # Attention visualization
│   ├── attention_heatmap_overlay.py
│   ├── interactive_heatmap.py
│   ├── attention_statistics.py
│   └── advanced_analysis.py       # Advanced analytics
├── popup/                          # Modal dialogs
│   ├── attention_analysis_popup.py
│   └── help_popup.py
├── utils/                          # Utility functions
│   ├── model_scanner.py           # Scan local models
│   └── dataset_scanner.py         # Scan local datasets
├── assets/                         # Static assets (CSS, images)
├── fonts/                          # Font files
├── LocalModels/                    # Downloaded models (gitignored)
└── LocalDatasets/                  # Downloaded datasets (gitignored)
```

---

## 🔬 Advanced Features

### Attention Extraction

ChartViz extracts cross-attention weights to show how the model connects text tokens to image regions:

```python
# Extract attention for analysis
attention_output = model.extract_attention(image, question)

# Access attention components
cross_attention = attention_output.cross_attention  # Text-to-image
text_self_attention = attention_output.text_self_attention
image_self_attention = attention_output.image_self_attention
```

### Debugging Model Behavior

Use attention visualizations to:
- ✅ Verify model focuses on relevant chart elements
- ✅ Identify when model misinterprets questions
- ✅ Detect attention anomalies (diffused, missing, etc.)
- ✅ Compare different models' reasoning patterns

### Statistical Analysis

**Attention Metrics**:
- **Entropy**: Measures attention spread (4-7 is good)
- **Gini Coefficient**: Measures concentration (0.3-0.6 is good)
- **Sparsity**: Fraction of near-zero weights (0.5-0.8 is good)
- **Max Attention**: Peak attention strength

**Interpretation**:
- High entropy → Model uncertain, attention diffused
- Low entropy → Model confident, focused attention
- High Gini → Over-focusing on small region
- Low Gini → Uniform attention, not selective

---

## 📞 Support

If you encounter issues:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review error messages in the console
3. Open an issue on GitHub with:
   - Error message and stack trace
   - System information (GPU, Python version)
   - Steps to reproduce

---

## 🗺️ Roadmap

- [ ] Support for more models (GPT-4V, Claude Vision, etc.)
- [ ] Batch processing for multiple charts
- [ ] Export analysis reports (PDF, HTML)
- [ ] Fine-tuning interface for custom datasets
- [ ] API endpoint for programmatic access
- [ ] Docker containerization
- [ ] Cloud deployment guides

---

**Made with ❤️ for the AI and Visualization community** at IIT Delhi **
