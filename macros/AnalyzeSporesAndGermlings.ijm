// Run folder, passed in by the app as a JVM property
runDir = call("java.lang.System.getProperty", "roi.runDir");
if (runDir == "null" || runDir == "") exit("The roi.runDir property is not set.");
if (!endsWith(runDir, File.separator)) runDir = runDir + File.separator;

// Enhance contrast for inconsistent images
run("Enhance Contrast...", "saturated=0.35 normalize");

// Crop image to 1000x1000 in center of image
size = 1000;
makeRectangle(getWidth()/2-size/2, getHeight()/2-size/2, size, size);
run("Crop");

// Subtract background from image
run("Subtract Background...", "rolling=10 light");

// Set thresholds
setAutoThreshold("MaxEntropy");
// setThreshold(0, 240, "raw");

// Black ROIs on a light background
setOption("BlackBackground", false);

// Generate a binary image from our image
run("Convert to Mask");

// Remove some of the small speckles
run("Despeckle");

// Try to fill in some of the gaps
run("Fill Holes");

// Generate ROIs
run("Set Measurements...", "area centroid perimeter fit shape feret's redirect=None decimal=3");
// Clear the shared Results table so each image's CSV holds only its own ROIs
run("Clear Results");
run("Analyze Particles...", "size=20-440 circularity=0.00-1.00 show=Overlay exclude include add");
roiManager("Show None");
saveAs("Results", runDir + File.getNameWithoutExtension(getTitle()) + ".csv");
