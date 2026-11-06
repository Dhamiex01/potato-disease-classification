# Potato Disease Classification (CNN)

A convolutional neural network built with TensorFlow/Keras that classifies potato leaf images into disease categories using the PlantVillage dataset.

## Overview

- **Task:** image classification (3 classes)
- **Framework:** TensorFlow / Keras
- **Input:** 256 x 256 RGB leaf images
- **Model:** 6 convolution + max-pooling blocks, followed by a dense classifier with softmax output
- **Pipeline:** resize and rescale, data augmentation (random flip and rotation), 80/10/10 train/validation/test split, cache/shuffle/prefetch for speed

## Project structure

```
potato-disease-classification/
├── potato_training.ipynb   # data loading, training, evaluation, prediction, model export
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The dataset is not included in this repository because of its size.

1. Download the PlantVillage dataset (the potato classes) from [Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village).
2. Place the class folders inside a folder named `PlantVillage/` next to the notebook:

```
PlantVillage/
├── Potato___Early_blight/
├── Potato___Late_blight/
└── Potato___healthy/
```

## Getting started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/potato-disease-classification.git
cd potato-disease-classification

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook potato_training.ipynb
```

Create an empty `models/` folder before running the last cell, which exports the trained model there as a versioned SavedModel (`models/1`, `models/2`, ...).

## Results

Add your final test accuracy here, for example:

| Metric | Value |
|--------|-------|
| Test accuracy | _xx.xx%_ |
| Test loss | _x.xxxx_ |

You can also add a screenshot of the training curves from the notebook.

## Notes

- Training on a CPU can be slow and memory-heavy. If the kernel crashes, reduce `BATCH_SIZE` or `IMAGE_SIZE` in the notebook.
- The notebook is commented step by step to explain each stage.

## Author

**Dhamiex'** - Electrical and Electronics Engineering, FUNAAB

## License

MIT (add a `LICENSE` file if you want to use this license).
