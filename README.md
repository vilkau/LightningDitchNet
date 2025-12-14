# LightningDitchNet
This repository provides a machine learning pipeline with command-line interfaces for preprocessing, training, evaluation, 
and inference to detect ditches from digital elevation models (DEMs) using a U-Net–based semantic segmentation approach.

---

## Project context
The code and documentation in this repository were developed as part of the Aalto University course 
GIS-E6010 – Project Course, carried out in collaboration with the Finnish Forest Centre (Metsäkeskus).

The overall project aimed to develop an open, production-ready method for automated detection of ditches in forests and 
peatlands using remote sensing data. This repository contains the machine learning pipeline and command-line interfaces 
developed for that project.

The full project repository (including graphical user interface and other components) is available here:  
https://github.com/klimesm/GIS_E6010_Project_Course_2025

---

## Repository structure

```
LightningDitchNet/
│
├── Documentation/                      # Module-level documentation for all pipeline stages
│   ├── inference_documentations.md
│   ├── model_documentations.md
│   ├── preprocessing_documentations.md
│   ├── test_documentations.md
│   ├── tools_documentations.md
│   └── train_documentations.md
│
├── env/
│   └── ditchnet_pytorch.yml            # Conda environment specification for the full pipeline
│
├── src/                                # Main source code
│   ├── utils/
│   │   ├── cli_args/                   # Command-line argument definitions for all pipeline stages
│   │   │   ├── inference_args.py       
│   │   │   ├── model_args.py           
│   │   │   ├── preprocessing_args.py   
│   │   │   ├── test_args.py            
│   │   │   └── training_args.py        
│   │   │
│   │   ├── config.py                   # Central configuration definitions
│   │   └── tools.py                    # Terrain feature generation and shared utilities
│   │
│   ├── inference.py                    # Run inference on DEM tiles
│   ├── model.py                        # Model definition (LightningDitchNet)
│   ├── preprocessing.py                # DEM preprocessing and chip generation
│   ├── test.py                         # Model evaluation
│   └── train.py                        # Model training script
│
├── .gitignore
├── LICENCE.md
└── README.md
```

---

## Research background
This work is inspired by and based on the methodology presented in the study:

> **Lidberg, W. et al. (2023).**  
> *Mapping Drainage Ditches in Forested Landscapes Using Deep Learning and Aerial Laser Scanning.*  
> Journal of Irrigation and Drainage Engineering, 149(3).  
> https://doi.org/10.1061/JIDEDH.IRENG-9796

This repository provides an independent and further developed implementation of the original approach. 

Key differences and adaptations include:

- Use of a PyTorch Lightning–based U-Net implementation instead of the original research framework.
- Extension of the input features from the single HPMF layer used in the original study to a two-channel input
combining High-Pass Median Filter (HPMF) and Impoundment Size Index (ISI) layers.
- Default use of an EfficientNet-B4 encoder in place of the Xception encoder used in the original study, 
with support for alternative encoder choices.
- Support for model ensembling during inference.

---

## Example use case

As an example use case, the pipeline was applied to train a ditch detection model using openly available reference data.

### Data

Training was performed using the following openly available dataset:

> Ågren, A. M., Paul, S. S., & Lidberg, W. (2022).  
> *Mapped drainage ditches in forested landscapes.*  
> Mendeley Data, V1.  
> https://doi.org/10.17632/zxkg43jsx8.1

### Results

The trained model achieved the following performance on the validation set:

- **Recall:** 87.28%
- **Precision:** 82.55%
- **Matthews Correlation Coefficient (MCC):** 0.846

These results illustrate the performance obtained using this pipeline and are dependent on the specific training 
configuration and dataset.

---

## Author
All code and documentation in this repository were developed by Ville Kauppinen 
as part of the Aalto University course GIS-E6010 – Project Course.

Email: ville.1.kauppinen@aalto.fi

---

## License
This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for more details.

---
