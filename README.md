# gmm_color_segmentation
UPenn ESE 650 Learning in Robotics, Project 1: Color Segmentation Based on Gaussian Mixture Model
By Wudao Ling

1. Put training images into folder .\roipoly_annotate\train  
   Put testing images into folder .\roipoly_annotate\test
2. Execute annotate.py to hand-label RedBarrel and TrickyCases.   
   Remember to change colorClass in the code between. TrickyCases are basically light yellow ceiling, windows and dark red floor.  
3. Run train.py to get a GMM Model  
4. Once the model is obtained, run predict.py to detect red barrel.   
   A image with bounding box and centroid will appear, also info like centroid, distance and numbers will be at command window. 
