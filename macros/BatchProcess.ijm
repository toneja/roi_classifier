// Macro to batch process a folder of images one at a time
inputFolder = getArgument();

// Main macro path, passed in by the app as a JVM property
mainMacro = call("java.lang.System.getProperty", "roi.mainMacro");
if (mainMacro == "null" || mainMacro == "") exit("The roi.mainMacro property is not set.");

// Get a list of files in the input folder
list = getFileList(inputFolder);

// Process each image file in the folder
for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], ".jpg")) {
        // Open the image
        open(inputFolder + "/" + list[i]);

        // Execute our main macro
        runMacro(mainMacro);

        // Close the image
        close();
    }
}

// Exit ImageJ
run("Quit");
