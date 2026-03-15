# AI Content Detector

A powerful Python web application for detecting AI-generated content in English academic papers and essays with Turnitin-level accuracy.

## Features

- **Multi-Model Ensemble**: Uses multiple transformer models (RoBERTa, DistilBERT) for higher accuracy
- **Linguistic Analysis**: Analyzes vocabulary diversity, sentence structure, and writing patterns
- **Fast Processing**: Optimized for quick analysis of academic papers
- **File Support**: Accepts TXT, DOCX, and PDF files
- **Interactive Dashboard**: Real-time visualization of results
- **Confidence Scoring**: Provides confidence levels for detection results

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-content-detector
```

2. Create a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Either:
   - Paste your essay/paper text directly into the text area
   - Upload a TXT, DOCX, or PDF file

4. Click "Analyze Content" to get results

## Detection Methodology

### Ensemble Approach
The application uses multiple models to improve accuracy:
- **RoBERTa**: For deep semantic understanding
- **DistilBERT**: For faster processing with good accuracy
- **Linguistic Features**: Handcrafted features for pattern recognition

### Analyzed Features
- Average sentence length
- Vocabulary diversity
- Punctuation patterns
- Word repetition scores
- Semantic consistency

### Accuracy
- Combines multiple models for Turnitin-level accuracy
- Provides confidence scores (high, medium, low)
- Cross-model validation for reliable results

## API Endpoints

- `GET /`: Main web interface
- `POST /analyze`: Analyze text for AI content
- `GET /health`: Check system health and loaded models

## File Upload Support

- **TXT**: Plain text files
- **DOCX**: Microsoft Word documents (requires python-docx)
- **PDF**: PDF documents (requires PyPDF2)

Maximum file size: 16MB

## Performance

- Processing time: < 5 seconds for typical essays
- Memory usage: Optimized for standard hardware
- GPU acceleration: Automatically uses CUDA if available

## Requirements

- Python 3.8+
- 4GB+ RAM recommended
- Internet connection for initial model download

## Dependencies

See `requirements.txt` for the complete list of dependencies.

## Configuration

The application automatically configures:
- Model loading (with fallbacks if models fail to load)
- GPU/CPU detection
- File upload limits
- Temporary file handling

## Limitations

- Works best with English text
- Minimum 50 characters required for reliable analysis
- Very short texts may have lower confidence scores

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please create an issue in the repository.

---

**Note**: This tool is designed to assist in maintaining academic integrity. Results should be used as part of a comprehensive review process.
