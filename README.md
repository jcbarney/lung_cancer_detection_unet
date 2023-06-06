# lung_cancer_detection_unet
U-Net model for detecting lung cancer nodules from CT scans

How to train models and test results:
1. Download data files from locations as shown in paper.pdf
2. Run train_unet_nodule_detection (may need to run in Google's Vertex AI or cloud platform of your choice as the U-Net architecture and data size are very large)
3. Run malignancy_model_train_test and see printed results
