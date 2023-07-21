These are the components working:
### Components
1. Patch Generation (data/patches/)
2. Predictions
    1. Localization Prediction
    2. Model Prediction
    3. Destruction Classification Mask (loc*des & hot encode)
3. Overlay Generation
    1. Localization Overlay
    2. Destruction Overlay
4. UnPatch & Tiff Mapping
    1. Masks
    2. Overlays