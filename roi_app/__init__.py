"""ROI Classifier: classify ImageJ particle ROIs with logistic regression.

Modules
-------
config       paths, defaults and constants
util         shared helpers (log type, timestamps)
state        settings remembered between sessions
settings     parsing and checking of the GUI's text fields
labels       class names: typed in the app, class_labels.txt, or generic
imagej       ImageJ download and batch runs (cancel and time limit supported)
datasets     labelled model datasets and per-image ROI tables
modelling    forward feature selection, evaluation, ROI classification
store        saved models, one per dataset, checked against the dataset's SHA-256
tables       sortable results table
dialogs      class-label dialog
app          the main window and background job handling
"""
