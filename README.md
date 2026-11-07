# ROI Classifier

A desktop app that classifies particles (spores, germlings and other ROIs) found by [ImageJ](https://imagej.net/software/imagej/) in fresh microscope images. It trains a logistic regression model on a labelled dataset, picks the most useful measurements with forward feature selection, then runs ImageJ over a folder of new images and reports, for each image, how many ROIs fall into each class.

## What it does

1. **Batch-runs ImageJ** over a folder of JPG images. ImageJ is downloaded automatically into `data/ImageJ/` the first time it is needed.
2. **Loads a labelled dataset**: an ImageJ Analyze Particles table (CSV) with an integer class column.
3. **Selects features and trains** a logistic regression model (scikit-learn pipeline with standard scaling) using forward Sequential Feature Selection.
4. **Classifies every ROI** in the new images and reports, per image, the total number of ROIs and the count and percentage for each class.

## Requirements

- Python 3.9 or newer
- Python packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`

  ```bash
  pip install numpy pandas scikit-learn matplotlib joblib
  ```

- Tkinter. It ships with most Python installers. On Linux, install it from your package manager, for example `sudo apt install python3-tk`.
- Java. The Windows, macOS and Linux (x86-64) ImageJ bundles include a Java runtime. For other systems, install Java 8 or newer.

ImageJ itself is not required in advance. The app downloads ImageJ 1.54 when it first runs a classification.

## Getting started

```bash
python roi_classifier_gui.py
```

1. **Model dataset (CSV):** choose the labelled dataset.
2. **Image folder:** choose the folder of new JPG images to classify.
3. **Target column:** the column holding the integer class labels (default `class`).
4. **Class labels:** name each class (see [Class labels](#class-labels)).
5. Run the actions in order:
   - **A. Test accuracy** splits the dataset, selects features on the training part only, and shows a classification report and ROC curve.
   - **B. Select features & (re)train** trains on the full dataset and saves the model. Run this once per dataset, and again whenever the dataset file changes.
   - **C. Classify images** runs ImageJ over the image folder and shows the per-image counts. This button can be cancelled while ImageJ is running.

The app remembers your inputs between sessions.

## Input dataset

The model dataset is a CSV in the format ImageJ writes for Analyze Particles, with one row per labelled ROI.

- The target column (default `class`) must hold whole numbers. Each class needs at least 2 ROIs.
- Feature columns are used when present. Missing ones are reported in the log and skipped. The features considered are:

  `Area`, `Perim.`, `Major`, `Minor`, `Circ.`, `Feret`, `MinFeret`, `AR`, `Round`, `Solidity`

- Rows with missing feature values are dropped, and the log says how many.
- The number of features to keep is set in the window (default 8). Eight gave the best results in testing.

## Class labels

Class numbers can be named in two ways:

- **In the app:** click **Edit class labels...** after loading a dataset. The dialog lists the classes found in the dataset.
- **In a file:** create `class_labels.txt` in the project folder. It holds a Python dictionary:

  ```python
  { -1: "debris", 0: "ungerminated", 1: "germinated" }
  ```

Names typed in the app take priority over the file. If neither is given, classes are shown as `class 0`, `class 1` and so on. The names must match the classes in the dataset exactly, or the run stops with a message listing the difference.

**Negative classes** (such as `-1` for debris) are treated as excluded. Their counts are still reported, marked `(excluded)`, but they are left out of each image's total and of every percentage.

## Outputs

| Location | Contents |
| --- | --- |
| `data/reports/` | Classification report (`.txt`) and ROC curve (`.png`) from each **A** run |
| `data/models/` | Saved model for each dataset (`.joblib`) and its JSON summary |
| `data/runs/<timestamp>/` | For each **C** run: the ROI tables from ImageJ, `imagej.log`, `per_image_counts.csv` and `classification_report.txt` |

The **ROI counts per image** tab shows the same per-image table. Click any column heading to sort it; click again to reverse the order.

Each saved model is checked against the exact dataset file it was trained on. If the dataset has changed since, the app asks you to run **B** again.

## Project layout

```
roi_classifier_gui.py         entry point: python roi_classifier_gui.py
roi_app/                      the application package
  config.py                   paths and constants
  app.py                      the main window and background jobs
  imagej.py                   ImageJ download and batch runs
  datasets.py                 model datasets and per-image ROI tables
  modelling.py                feature selection, evaluation and classification
  store.py                    saved models
  labels.py                   class names
  settings.py, state.py       input checks and remembered settings
  tables.py, dialogs.py       sortable table and class-label dialog
macros/
  BatchProcess.ijm            ImageJ batch macro: runs the main macro on each image
  AnalyzeSporesAndGermlings.ijm   ImageJ main macro: preprocessing, particle analysis, CSV output
class_labels.txt              optional class names (see above)
data/                         everything the app generates (created automatically)
```

## How ImageJ is driven

The app starts ImageJ in batch mode:

```
java -Droi.runDir=<run folder> -Droi.mainMacro=<main macro> \
     -Droi.sizeMin=<min> -Droi.sizeMax=<max> \
     -jar ij.jar -batch BatchProcess.ijm "<image folder>"
```

- The image folder is the batch macro's single argument.
- The run folder, the main macro's path and the particle size bounds are passed as JVM properties. Macros read them with `call("java.lang.System.getProperty", ...)`.
- The batch macro runs the main macro on each `.jpg` image. The main macro writes one results CSV per image into the run folder, and the app collects them from there.

Particle size bounds (default 50 to 500 ImageJ units) are set in the window.

## Notes

- All files the app creates or downloads stay inside the project folder, under `data/`.
- Closing the window while ImageJ is running stops ImageJ.
- Log lines from ImageJ are filtered to warnings and errors. The full ImageJ output is kept in `imagej.log` for each run.
