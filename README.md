# The RV-PBS dataset

This repository hosts the dataset created for the paper titled ***“Instance Segmentation of Peripheral Blood Smear and Refining Classification via Domain Adaptation“.***

The dataset is annotated using [CVAT](https://github.com/openvinotoolkit/cvat). We are planning to release an extended version of this dataset soon. If you are a haematologist, then you could help us by annotating and adding more data. **Please make sure that the data is ethically cleared before uploading new data in public servers, such as Github.**

### Data creation process

For this study, we have created a novel WBC dataset comprising 10 classes known as the ramakirshna vivekananda peripheral blood smear (RV-PBS) dataset. Air dried peripheral blood smears are stained by Leishman stain following standard protocol and examined under an oil immersion lens using 10X eyepiece magnification (final magnification–1000X). Photographs taken by iPhone XR 12-megapixel camera with f/1.8 aperture. The dataset comprises high resolution (4032 x 3024) images of blood smear slides.

### Snapshot of dataset creation using CVAT 

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/cvat_mask_basophil.png">
</center>

## Some relevant stuffs from the paper

**Please study the paper for getting more insights. Here are some snapshots from the paper:**

### Smear slides cropped dataset

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/1_Smear_Slides_8_cropped.png">
</center>

### Schematic diagram for extraction of cells ready to be sent to domain adaptation pipeline

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/2_1_Pipeline_general.png">
</center>

### Classification model used with different backbones

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/2_2_Classification_pipeline.png">
</center>

### Results Table

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/3_8_common_classes_results.png">
</center>

### Results Table

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/4_pretrained_pipeline.png">
</center>

### Results Table

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/5_1_Smear_Slides_8_pretrained.png">
</center>

### Final output of the detection and segmentation pipeline for MaskRCNN and Domain Adaptation

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/5_2_MaskRCNN_DA.png">
</center>

### Mask R-CNN losses

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/5_3_MaskRCNN_losses.png">
</center>

### Domain Adaptation models

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/6_DA_model.png">
</center>

### Results Table for Domain Adaptation

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/7_DA_results.png">
</center>

### Domain Adaptation losses

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/8_DA_metrics.png">
</center>

### Full pipeline

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/9_DA_pipeline.png">
</center>

### JSON outputs which can be used for automated annotation of new slides (Future work)

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/10_json_outputs.png">
</center>


## If you find this work useful, please consider citing

```

```
