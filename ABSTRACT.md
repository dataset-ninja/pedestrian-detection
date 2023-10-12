Pedestrian Detection is a subfield of object detection that is necessary for several applications such as person tracking, intelligent surveillance system, abnormal scene detection, and intelligent cars. We prepared a dataset for addressing the false positives that occur during the person detection process. Some objects have very similar features to those of a person. If a model is trained using a dataset containing only persons, it leads to several false positives since it cannot differentiate such objects from that of a person. Our dataset includes person and person-like objects (PnPLO). Person-like objects that we introduce in our dataset are statues, mannequins, scarecrows, and robots. We used the SSD model to show that, on training a model using our dataset, we can significantly reduce the false positives during detection when compared to models trained on standard person datasets, thereby improving the precision. The dataset consists of 944 training images, 160 validation images, and 235 images for testing, with a total of 1626 person and 1368 nonhuman labelling.


<!-- Sample image template:
<img src="https://github.com/dataset-ninja/gland-segmentation/assets/78355358/f158d0dd-71d5-41a2-aba5-4a5f57d54c35" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Image description.</span> -->
