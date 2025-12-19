# From Nine to One: Combining ICF Functioning Level Classifiers in the A-PROOF Project

This repository contains all codes for predicting functioning levels from clinical notes, developed as part of **Urtė Jakubauskaitė's Master's thesis**.  

> **Note:** The medical dataset is not publicly available due to privacy and ethical restrictions. A PDF copy of the thesis is included.  

## Repository Structure
```
├── README.md
├── requirements.txt
├── Urte_Jakubauskaite_2025_MA_Linguistics_Text_Mining_Thesis.pdf
└── src
    ├── dataset_creation
    │   └── Scripts for generating new datasets and further splitting them into training and development files
    ├── error_analysis
    │   └── Scripts for performing error analyses on model predictions as well as the file with words with target linguistic elements.
    ├── statistics
    │   └── Scripts to compute dataset statistics
    └── models_and_evaluation
        ├── training
        │   └── Scripts for training the models used in the study
        └── evaluation
            ├── note_level
            │   └── A script to evaluate models on the note level
            └── sentence_level
                └── Scripts to evaluate models on the sentence level
```

> Each script includes a docstring explaining its functionality, input parameters, and outputs.
