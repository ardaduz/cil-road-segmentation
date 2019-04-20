## TRAINING-PREDICTION DETAILS

* Trained with 100 competition-data images
* Test images are downscaled to 304x304 with LANCZOS interpolation, passed through pipeline
* Predictions upscaled to 608x608 with LANCZOS interpolation
* histogram_resolution = 256
* unary_lambda_value = 1e-20
* pairwise_sigma_value = 5
* window_size = 9
