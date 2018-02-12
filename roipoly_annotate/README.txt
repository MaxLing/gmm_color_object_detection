simple annotation program with roipoly:
allows for multiple polygons on same image

Run annotate.py to annotate images
put images into the "images/" subfolder
change the "colorClass" variable to indicate which color you will be annotating, and make a folder with that name under "labeled_data"

this program will save the the binary masks indicating which pixels are part of the color class into the labeled_data/<colorClass> subfolder
this program will also only annotate images that have not already been annotated (it will check if a saved binary mask is already available). To redo an image, simply delete the saved mask from the subfolder


