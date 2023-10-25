# Image Retrieval with Siamese Networks

# To do:
- [ ] Keep queries out of the training
## Data preprocessing
- [ ] Function that takes image and returns vector (composed by keypoints)
- [ ] Function that goes through the matrix (Dataset) and generates image pair in `X_data` and puts the score in `Y_data` (both are .npy arrays).
- [ ] Split DataSet in Train, Test, Validation with scikit learn function. 80/10/10
## Build Model
- [ ] Build Siamese Network. Try different architectures (ex: Euclidian_dist+sigmoid).
## Train Model
- [ ] Train model. Tune parameters.
## Evaluate Model
- [ ] Evaluate model. Accuracy, Precision, Recall, F1-Score.
## Compare with other methods
- [ ] Compare performance with other methods like CNNs ot Bag Of Words.
## Write Report
- [ ] Write the report
