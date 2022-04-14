### How did we train models with different fractions of data?

After cloning [the project](https://github.com/MohitLamba94/Restoring-Extremely-Dark-Images-In-Real-Time), we had to modify `train_test_ours\train_test_SID_CVPR_18\train.py` file in order to train the original model on fractions of data. 
Additionally, we had to create 4 different directories in `train_test_ours\train_test_SID_CVPR_18` with the original SID training and test images:

- long_testing
- short_testing
- long_train_validation
- short_train_validation

These directories contained short-exposed images and long-exposed images from the original dataset. In the end, this is how we managed to train 4 different models on Google Cloud, by just modifying
`data_fraction` variable in `train.py` (we set it to 0.4 by default).
