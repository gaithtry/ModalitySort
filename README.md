# ModalitySort
Medical Image Modality Sorter 

ModalitySort is a deep learning tool for automatic neuroimaging modality classification. It classifies NIfTI brain images into six categories (ncCT, CTA, MRA, T1w, T2w, and FLAIR), serving as a quick preprocessing step before modality-specific imaging pipelines.

Best model weights can be downloaded from here (best_metric_model.pth): 
[https://drive.google.com/file/d/1vlnBCbZ_Y7k4yYlHJxxqp4M4IUKM2ssV/view?usp=sharing](https://drive.google.com/file/d/1YucpGAeaG9aXMkKlN7RMcteHTxuDrXcp/view?usp=sharing)

Install requirements:
```
pip install -r requirements.txt
```

For prediction, run any NIfTI image into prediction model with saved weights:
```
python predict_modality.py <NIfTI image.nii.gz>
```

For detailed breakdown of prediction on per-slice basis:
```
python prediction_per_slice.py <NIfTI image.nii.gz>
```

To view model's detailed breakdown of model validation performance:
```
python confusion_matrix_v2.py
```

To train from scratch use:
```
python training.py
```
