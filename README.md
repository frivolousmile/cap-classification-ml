# Image-Based Cap Detection Using Machine Learning

## Project
Binary image classification: **Cap On (1)** vs **Cap Off (0)**.

## Training concepts demonstrated
- OpenCV image loading and resizing
- NumPy array conversion
- Flattening image pixels into ML features
- Stratified train/test split
- K-Nearest Neighbors
- Support Vector Machine (linear and RBF kernels)
- 5-fold cross-validation
- GridSearchCV hyperparameter tuning
- Confusion matrix and classification metrics
- Optional Streamlit deployment

## Dataset
This project uses the public `ravee360/Cap-detection` repository as a source for a cap/no-cap image dataset. The repository describes 671 cap-wearing images and 529 non-cap-wearing images (1,200 total) and contains a split/images and split/annotations structure.

Source:
https://github.com/ravee360/Cap-detection

## Run
1. Install Python 3.10+.
2. `pip install -r requirements.txt`
3. `python download_dataset.py`
4. `python train_model.py`
5. Review `outputs/results.json`, `outputs/confusion_matrix.png`, and `outputs/accuracy_comparison.png`.
6. Optional: `streamlit run app.py`

## Important
The report must use the metrics produced by `train_model.py`. Do not replace them with invented values.
If the dataset's annotation class mapping differs, adjust `CAP_CLASS_ID` in `train_model.py` after checking an annotation file.
