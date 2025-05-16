# Face Verification Description

The aim of this task is to implement a system for verifying human faces in a selected programming language (Python).
The solution uses the FaceVerification dataset, which contains pairs of images with information about whether they belong to the same person. 
The project is divided into two parts: 

Part 1 – classic face verification: The dataset is loaded together with the corresponding CSV file. An analysis of the number of identical and different pairs is performed, 
  including visualization of representative cases. Using the SIFT algorithm, three selected face samples (same and different people) are compared at the keypoint level. 
  Two feature extraction methods are implemented: ResNet50 trained on ImageNet, ArcFace (via DeepFace) trained on human faces. 
  The feature comparison is performed using a recommended metric (e.g. L2 or cosine distance). The results are visualized using a ROC curve, the AUC is calculated, and the optimal threshold is determined. 
  Finally, a visual analysis of the best and worst cases for True/False pairs is performed. 
  
Part 2 – verification using triplet loss: Based on the results from Part 1, the best performing network (ArcFace) is selected. 
  A triplet structure is created from the dataset: Anchor–Positive and Anchor–Negative. The comparison is performed again using ArcFace and the results are evaluated using a ROC curve and a confusion matrix. 
  The modified dataset in the form of triplets is saved to CSV for further use. This project connects classical and modern approaches to face recognition and allows for a practical comparison of the performance of different models in the verification task.
