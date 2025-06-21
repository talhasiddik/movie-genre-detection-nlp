# Movie Genre Detection from YouTube Trailers

This repository contains code and documentation for a movie genre detection system that uses YouTube trailers as input. The approach utilizes multimodal deep learning to analyze visual and audio content from trailers.

## Project Overview

This project demonstrates how to:
- Determine a movie genre label codebook based on MovieLens data
- Label a large-scale dataset of movie trailers
- Implement a multimodal approach for genre detection
- Process both visual frames and audio content from trailers
- Evaluate model performance with appropriate metrics

## Repository Structure

- `movie_genre_detection.ipynb`: Jupyter notebook containing the full implementation
- `experimental_design.md`: Detailed description of the experimental approach
- `requirements.txt`: Python dependencies for the project
- `LICENSE`: MIT License file
- `CITATION.cff`: Citation file in Citation File Format for easy referencing

## Getting Started

### Setting up a Virtual Environment

1. Create a Python 3.11 virtual environment:

```bash
py -3.11 -m venv .venv
```

2. Activate the virtual environment:

- Windows:
```bash
.venv\Scripts\activate
```
- Unix/MacOS:
```bash
source .venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure you have the MovieLens 20M dataset downloaded in the `ml-20m` directory
2. Run the Jupyter notebook to train and evaluate the model:

```bash
jupyter notebook movie_genre_detection.ipynb
```

## Data Requirements

- MovieLens 20M dataset: https://grouplens.org/datasets/movielens/20m/
- YouTube trailers linked to the MovieLens IDs

## Citation

If you use this code or methodology in your research, please cite this work:

```bibtex
@misc{movie_genre_detection_2024,
  title={Movie Genre Detection from YouTube Trailers: A Multimodal Deep Learning Approach},
  author={Talha Siddique},
  year={2024},
  publisher={GitHub},
  url={https://github.com/talhasiddik/movie-genre-detection-nlp}
}
```

You can also use the `CITATION.cff` file in this repository, which follows the Citation File Format standard and can be automatically processed by GitHub and other platforms.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project is intended for educational and research purposes.

## Acknowledgements

- MovieLens dataset from GroupLens Research
- The project design is informed by research in multimodal deep learning and video understanding
- Special thanks to the open-source community for the tools and libraries used in this project
