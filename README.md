# From Nine to One: Combining ICF Functioning Level Classifiers in the A-PROOF Project

This repository contains all codes used to predict functioning levels from clinical notes, developed as part of **Urtė Jakubauskaitė's Master's thesis**.  

> **Note:** The medical dataset is not publicly available due to privacy and ethical restrictions. A PDF copy of the thesis is included.  

## Repository Structure
```
├── README.md
├── requirements.txt
├── Urte_Jakubauskaite_2025_MA_Linguistics_Text_Mining_Thesis.pdf
└── src
    ├── dataset_creation
    │   └── Scripts for generating new datasets and further splitting them into training and development files.
    ├── error_analysis
    │   ├── FAC_analysis
    │   │   └── Scripts for performing error analysis on model predictions for the Walking (FAC) category.
    │   └── linguistic_analysis
    │       └── Scripts for performing error analysis on model predictions for negations, minimizers, and intensifiers, along with a file containing linguistic elements.
    ├── statistics
    │   └── Scripts for computing statistics of multiple datasets.
    └── models_and_evaluation
        ├── training
        │   └── Scripts for training the models.
        └── evaluation
            ├── note_level
            │   └── A script to evaluate models on the note level.
            └── sentence_level
                └── Scripts to evaluate models on the sentence level.
```

> Each script includes a docstring explaining its functionality.
