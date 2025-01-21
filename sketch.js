/**
 * Image Processing Application
 *
 * Note: This assignment was tested using the Firefox browser. For optimal performance,
 * it is recommended to use Firefox during testing. Thank you!
 *
 * Reflections:
 * This assignment was an enjoyable and rewarding experience. I implemented several features
 * successfully and overcame challenges during development, ultimately learning a lot in the process.
 *
 * Tasks Overview:
 *
 * Task 8: Segmentation for Each Color Channel
 * - The segmentation results vary for each channel (red, green, blue) because each channel represents
 *   a specific color component.
 * - Slider Behavior:
 *   - Starting position: Displays the same image for all channels.
 *   - End position: All channels turn black.
 * - Threshold sliders control segmentation intensity, allowing users to highlight or isolate specific
 *   regions in each channel.
 * - Insights:
 *   - Enables selective emphasis on color characteristics.
 *   - Provides flexibility for image analysis and processing.
 *
 * Task 11: Segmentation in Different Color Spaces
 * - Segmenting images in various color spaces (HSV, Y'CbCr, RGB) produces distinct results:
 *   - **HSV**:
 *     - Hue: Represents color type.
 *     - Saturation: Indicates intensity.
 *     - Value: Reflects brightness.
 *     - Observation: Segmentation is sensitive to saturation; low-saturation regions are more affected.
 *   - **Y'CbCr**:
 *     - Separates luminance (intensity) from chrominance (color).
 *     - Useful for distinguishing brightness and darkness.
 *     - Commonly used in video compression and selective segmentation tasks.
 *   - **RGB**:
 *     - Direct segmentation is effective but more prone to noise.
 *     - Complex lighting conditions and color variations can increase noise levels.
 * - Improving Segmentation:
 *   - Adjusting thresholds balances target color capture and noise filtering.
 *   - Fine-tuning parameters for each color space enhances segmentation results tailored to input data.
 *
 * Extension Features:
 * - **Custom Filters**:
 *   - Added a Y'CbCr filter and a merged filter beyond the assignment tasks.
 * - **Real-Time Face Detection and Filters**:
 *   - Used `ml5.js` library for real-time face detection and anime-style filters.
 *   - Transforms video input into dynamic overlays like anime or cat-themed visuals.
 *
 * Challenges:
 * 1. **Face Detection**:
 *    - Implemented using `ml5.js` faceApi for facial landmark recognition.
 *    - Used landmarks for accurate filter overlays.
 * 2. **Dynamic Scaling**:
 *    - Developed a scaling function to adjust filter sizes based on face size.
 *    - Calibrated for a natural appearance.
 *
 * Problems and Solutions:
 * - **Performance**:
 *   - Real-time video processing required optimizing drawing functions and efficient resource usage.
 * - **Detection Accuracy**:
 *   - Initial issues were resolved by tweaking model parameters and improving lighting conditions.
 *
 * Project Outcomes:
 * - Successfully integrated real-time detection with creative overlay effects.
 * - Future Work:
 *   - Explore improved detection techniques.
 *   - Add user-driven customization for greater flexibility.
 */

// Define video and screenshot variables
let video;
let screenshot;

// Define buttons
let loadBtn, saveBtn, snapshotBtn;

// Define RGB channels
let redChannel, greenChannel, blueChannel;

// Define segmented RGB channels
let segmentedRed, segmentedGreen, segmentedBlue;

// Define sliders for red, green, blue, HSV threshold, and YCbCr threshold
let redSlider,
  greenSlider,
  blueSlider,
  hsvThresholdSlider,
  ycbcrThresholdSlider;

// Define HSVConversion variable
let HSVConversion;

// Define YCbCr image variable
let ycbcrImg;

// Define face detection variables
let detector;
let classifier = objectdetect.frontalface;
let faces;
let applyEffect = false;

// Define faceapi and detections variables
let faceapi;
let detections = [];

function setup() {
  // Create a canvas with dimensions 600x1000 pixels
  createCanvas(600, 1000);

  // Set pixel density to 1 for consistency across different displays
  pixelDensity(1);

  // Create a video capture element
  video = createCapture(VIDEO);
  video.size(160, 120); // Set size of 160x120
  video.hide(); // Hide the video element initially

  // Create buttons for taking, saving, and loading screenshots
  setupButtons();

  // Create sliders for each channel
  setupSliders();

  // Initialize face detector with specified parameters
  detector = new objectdetect.detector(160, 120, 1.2, classifier);

  // Create an image element for face detection
  faceDectectImg = createImage(160, 120);

  // Options for faceapi detection
  const options = {
    withLandmarks: true,
    withDescriptors: false,
  };

  // Initialize FaceAPI with the video element, options, and callback function for model readiness
  faceapi = ml5.faceApi(video, options, modelReady);
}

function draw() {
  background(255);
  setupText();
  // Display webcam video
  image(video, 20, 0, 160, 120);

  let inputImage = screenshot || video; // Use the original input image
  let grayscaleImage = grayScale(inputImage);

  // Display grayscale image
  image(grayscaleImage, 200, 0, 160, 120);

  // Display color channel
  displayColorChannels(inputImage);

  // Display Red, Green, and Blue channels
  image(redChannel, 20, 140, 160, 120); // Red channel below webcam
  image(greenChannel, 200, 140, 160, 120); // Green channel below grayscaled image
  image(blueChannel, 380, 140, 160, 120); // Blue channel beside green channel

  // Display segmented color channels with sliders
  displaySegmentedColorChannels(inputImage);

  // Display Segmented Red, Green, and Blue channels
  image(segmentedRed, 20, 280, 160, 120);
  image(segmentedGreen, 200, 280, 160, 120);
  image(segmentedBlue, 380, 280, 160, 120);

  // diplay webcam a second time
  image(inputImage, 20, 435, 160, 120);

  // Colour Space HSV
  HSVConversion = convertRGBToHSV(inputImage);
  image(HSVConversion, 200, 435, 160, 120);

  // Colour Space Y'cbcr
  ycbcrImg = convertToYcbcr(inputImage);
  image(ycbcrImg, 380, 435, 160, 120);

  // Perform segmentation on HSV
  let thresholdHSV = hsvThresholdSlider.value();
  let segmentedHSV = segmentImage(HSVConversion, thresholdHSV);
  image(segmentedHSV, 200, 580, 160, 120);

  // Perform segmentation on Y'CbCr
  let thresholdYcbcr = ycbcrThresholdSlider.value();
  let segmentedYcbcr = segmentImage(ycbcrImg, thresholdYcbcr);
  image(segmentedYcbcr, 380, 580, 160, 120);

  // Face detection
  // Draw the original inputImage first
  image(inputImage, 20, 580, 160, 120);
  detectFaceFromImg(inputImage, 20, 550);

  // Anime Face Filter
  // Draw the original inputImage first
  image(inputImage, 200, 735, 160, 120);

  // Conditional check for detected faces
  if (detections.length > 0) {
    push();
    // Call function to draw anime filter with detected faces
    drawAnimeFilter(detections);
    pop();
  }
}

// Function to set up buttons for taking, saving, and loading screenshots
function setupButtons() {
  // Button to take a screenshot
  snapshotBtn = createButton("Take Screenshot");
  snapshotBtn.position(400, 10);
  snapshotBtn.mousePressed(takeScreenshot);

  // Button to save a screenshot
  saveBtn = createButton("Save Screenshot");
  saveBtn.position(400, 40);
  saveBtn.mousePressed(saveScreenshot);

  // Button to load a screenshot
  loadBtn = createFileInput(loadScreenshot);
  loadBtn.position(400, 70);
}

function setupText() {
  textSize(12); // Set text size
  textAlign(CENTER, TOP); // Align text to the center horizontally and to the top vertically
  text("Webcam", 100, 125);
  text("Grayscale + 20% brightness", 280, 125);
  text("Red Channel", 100, 265);
  text("Green Channel", 280, 265);
  text("Blue Channel", 460, 265);
  text("Segmented Red Channel", 100, 420);
  text("Segmented Green Channel", 280, 420);
  text("Segmented Blue Channel", 460, 420);
  text("Webcam (repeat)", 100, 565);
  text("RGB to HSV", 280, 565);
  text("RGB to Ycbcr", 460, 565);
  text("Face Detection", 100, 715);
  text("Segmented HSV", 280, 715);
  text("Segmented Ycbcr", 460, 715);
  text("'g' for grayscale + 20% brightness", 100, 760);
  text("'b' for blur,'h' for HSV, 'p' for pixelate,", 100, 780);
  text("'y' for YCbCr, 'm' for merged effect", 100, 800);
}

// Function to set up sliders for adjusting color channels and threshold values
function setupSliders() {
  // Red channel slider
  redSlider = createSlider(0, 255, 127);
  redSlider.position(26, 400);

  // Green channel slider
  greenSlider = createSlider(0, 255, 127);
  greenSlider.position(206, 400);

  // Blue channel slider
  blueSlider = createSlider(0, 255, 127);
  blueSlider.position(386, 400);

  // HSV threshold slider with initial value
  hsvThresholdSlider = createSlider(0, 255, 127);
  hsvThresholdSlider.position(206, 700);

  // YCbCr threshold slider with initial value
  ycbcrThresholdSlider = createSlider(0, 255, 127);
  ycbcrThresholdSlider.position(386, 700);
}

// Function to display segmented images on the canvas
function displaySegmentedImages(segmentedHSV, segmentedYcbcr) {
  // Display the HSV segmented image at specified position and size
  image(segmentedHSV, 200, 550, 160, 120);

  // Display the YCbCr segmented image at specified position and size
  image(segmentedYcbcr, 380, 550, 160, 120);
}

// Function to capture a screenshot from the webcam
function takeScreenshot() {
  // Capture the current frame and set its size to 160x120 pixels
  screenshot = createImage(video.width, video.height);
  screenshot.copy(
    video,
    0,
    0,
    video.width,
    video.height,
    0,
    0,
    video.width,
    video.height
  );
  screenshot.resize(160, 120);

  // Create an image object to hold the screenshot
  let img;
  img = createImage(screenshot.width, screenshot.height);
  img.copy(
    screenshot,
    0,
    0,
    screenshot.width,
    screenshot.height,
    0,
    0,
    screenshot.width,
    screenshot.height
  );

  // Use the ml5.faceApi to detect faces in the screenshot
  faceapi = ml5.faceApi(
    { withLandmarks: true, withDescriptors: false },
    function () {
      faceapi.detect(img, gotResults);
    }
  );
}

// Function to save the current screenshot
function saveScreenshot() {
  // Save the screenshot
  save(screenshot, "screenshot.png");
}

// Function to load a selected image file as a screenshot
function loadScreenshot(file) {
  // Check if a file is selected
  if (file.type === "image") {
    // Load the selected image and display it on the canvas
    screenshot = loadImage(file.data, () => {
      screenshot.resize(160, 120);
    });

    // Create an image object to hold the loaded screenshot
    let img;
    img = createImage(screenshot.width, screenshot.height);
    img.copy(
      screenshot,
      0,
      0,
      screenshot.width,
      screenshot.height,
      0,
      0,
      screenshot.width,
      screenshot.height
    );

    // Use the ml5.faceApi to detect faces in the loaded screenshot
    faceapi = ml5.faceApi(
      { withLandmarks: true, withDescriptors: false },
      function () {
        faceapi.detect(screenshot, gotResults);
      }
    );
  } else {
    console.error("Please select a valid image file.");
  }
}

// Function to convert an image to grayscale with increased brightness
// using the luma calculation and a + 20% brightness
function grayScale(img) {
  // Create a new image object with the same dimensions as the input image
  var editedImg = createImage(img.width, img.height);

  // Load pixels for both input and output images
  editedImg.loadPixels();
  img.loadPixels();

  // Iterate through each pixel in the input image
  for (var x = 0; x < img.width; x++) {
    for (var y = 0; y < img.height; y++) {
      // Calculate the pixel index in the image pixel array
      var index = (y * img.width + x) * 4;

      // Extract the red, green, and blue components of the pixel
      var pixRed = img.pixels[index + 0];
      var pixGreen = img.pixels[index + 1];
      var pixBlue = img.pixels[index + 2];

      // Convert to grayscale using luma calculation
      var gray = pixRed * 0.299 + pixGreen * 0.587 + pixBlue * 0.114;

      // Increase brightness by 20% without exceeding 255
      gray = min(255, gray * 1.2);

      // Set the grayscale values for the output image pixels
      editedImg.pixels[index + 0] = gray;
      editedImg.pixels[index + 1] = gray;
      editedImg.pixels[index + 2] = gray;
      editedImg.pixels[index + 3] = 255; // Set alpha value to fully opaque
    }
  }

  // Update the pixels for the output image
  editedImg.updatePixels();

  // Return the processed grayscale image
  return editedImg;
}

// Function to display the Red, Green, and Blue channels
function displayColorChannels(inputImage) {
  // Create image objects for Red, Green, and Blue channels
  redChannel = createImage(inputImage.width, inputImage.height);
  greenChannel = createImage(inputImage.width, inputImage.height);
  blueChannel = createImage(inputImage.width, inputImage.height);

  // Load pixels for input and output images
  inputImage.loadPixels();
  redChannel.loadPixels();
  greenChannel.loadPixels();
  blueChannel.loadPixels();

  // Iterate through each pixel in the input image
  for (let x = 0; x < inputImage.width; x++) {
    for (let y = 0; y < inputImage.height; y++) {
      // Calculate the pixel index in the input image pixel array
      let index = (y * inputImage.width + x) * 4;

      // Extract the red, green, and blue components of the pixel
      let pixRed = inputImage.pixels[index + 0];
      let pixGreen = inputImage.pixels[index + 1];
      let pixBlue = inputImage.pixels[index + 2];

      // Set values for Red channel, no green or blue components
      redChannel.pixels[index + 0] = pixRed;
      redChannel.pixels[index + 1] = 0;
      redChannel.pixels[index + 2] = 0;
      redChannel.pixels[index + 3] = 255; // Set alpha value to fully opaque

      // Set values for Green channel, no red or blue components
      greenChannel.pixels[index + 0] = 0;
      greenChannel.pixels[index + 1] = pixGreen;
      greenChannel.pixels[index + 2] = 0;
      greenChannel.pixels[index + 3] = 255;

      // Set values for Blue channel, no red or green components
      blueChannel.pixels[index + 0] = 0;
      blueChannel.pixels[index + 1] = 0;
      blueChannel.pixels[index + 2] = pixBlue;
      blueChannel.pixels[index + 3] = 255;
    }
  }

  // Update pixels for the output images
  redChannel.updatePixels();
  greenChannel.updatePixels();
  blueChannel.updatePixels();

  // Update slider values to their current positions
  redSlider.value(redSlider.value());
  greenSlider.value(greenSlider.value());
  blueSlider.value(blueSlider.value());
}

// Function to display segmented color channels based on slider values
function displaySegmentedColorChannels(inputImage) {
  // Apply slider values to color channels
  let redValue = redSlider.value();
  let greenValue = greenSlider.value();
  let blueValue = blueSlider.value();

  // Create segmented color channel images
  segmentedRed = createImage(inputImage.width, inputImage.height);
  segmentedGreen = createImage(inputImage.width, inputImage.height);
  segmentedBlue = createImage(inputImage.width, inputImage.height);

  // Load pixels for input and output images
  inputImage.loadPixels();
  segmentedRed.loadPixels();
  segmentedGreen.loadPixels();
  segmentedBlue.loadPixels();

  // Iterate through each pixel in the input image
  for (let x = 0; x < inputImage.width; x++) {
    for (let y = 0; y < inputImage.height; y++) {
      // Calculate the pixel index in the input image pixel array
      let index = (y * inputImage.width + x) * 4;

      // Extract the red, green, and blue components of the pixel
      let pixRed = inputImage.pixels[index + 0];
      let pixGreen = inputImage.pixels[index + 1];
      let pixBlue = inputImage.pixels[index + 2];

      // Create segmented images based on slider values
      // Segmented Red channel
      segmentedRed.pixels[index + 0] = pixRed > redValue ? pixRed : 0;
      segmentedRed.pixels[index + 1] = 0;
      segmentedRed.pixels[index + 2] = 0;
      segmentedRed.pixels[index + 3] = 255; // Set alpha value to fully opaque

      // Segmented Green channel
      segmentedGreen.pixels[index + 0] = 0;
      segmentedGreen.pixels[index + 1] = pixGreen > greenValue ? pixGreen : 0;
      segmentedGreen.pixels[index + 2] = 0;
      segmentedGreen.pixels[index + 3] = 255;

      // Segmented Blue channel
      segmentedBlue.pixels[index + 0] = 0;
      segmentedBlue.pixels[index + 1] = 0;
      segmentedBlue.pixels[index + 2] = pixBlue > blueValue ? pixBlue : 0;
      segmentedBlue.pixels[index + 3] = 255;
    }
  }

  // Update pixels for the segmented channel images
  segmentedRed.updatePixels();
  segmentedGreen.updatePixels();
  segmentedBlue.updatePixels();
}

// Function to convert an RGB image to HSV (Hue, Saturation, Value)
function convertRGBToHSV(img) {
  // Create a new image for HSV
  let hsvImage = createImage(img.width, img.height);
  hsvImage.loadPixels();
  img.loadPixels();

  // Iterate through each pixel in the input RGB image
  for (let x = 0; x < img.width; x++) {
    for (let y = 0; y < img.height; y++) {
      // Calculate the pixel index in the pixel array
      let index = (y * img.width + x) * 4;

      // Normalize RGB values to the range [0, 1]
      let pixRed = img.pixels[index + 0] / 255;
      let pixGreen = img.pixels[index + 1] / 255;
      let pixBlue = img.pixels[index + 2] / 255;

      // Find the maximum and minimum values from the RGB triplet
      let cMax = max(pixRed, max(pixGreen, pixBlue));
      let cMin = min(pixRed, min(pixGreen, pixBlue));

      // Calculate saturation (S)
      let sat = (cMax - cMin) / cMax;

      // Calculate value (V)
      let val = cMax;

      // Calculate normalized RGB values
      let rPrime = (cMax - pixRed) / (cMax - cMin);
      let gPrime = (cMax - pixGreen) / (cMax - cMin);
      let bPrime = (cMax - pixBlue) / (cMax - cMin);

      let hue;
      // Calculate hue
      if (sat == 0) {
        hue = 0;
      } else {
        if (pixRed == cMax && pixGreen == cMin) {
          hue = 5 + bPrime;
        } else if (pixRed == cMax && pixGreen != cMin) {
          hue = 1 - gPrime;
        } else if (pixGreen == cMax && pixBlue == cMin) {
          hue = rPrime + 1;
        } else if (pixGreen == cMax && pixBlue != cMin) {
          hue = 3 - bPrime;
        } else if (pixRed == cMax) {
          hue = 3 + gPrime;
        } else {
          hue = 5 - rPrime;
        }
        // Convert hue to degrees
        hue *= 60;
        // Ensure hue is within the range [0, 360)
        if (hue < 0) {
          hue += 360;
        }
      }

      // Convert HSV to the range [0, 255] and assign to the pixels
      hsvImage.pixels[index + 0] = (hue / 360) * 255; // H
      hsvImage.pixels[index + 1] = sat * 255; // S
      hsvImage.pixels[index + 2] = val * 255; // V
      hsvImage.pixels[index + 3] = 255; // Alpha
    }
  }

  // Update the pixel array for the HSV image
  hsvImage.updatePixels();
  // Return the resulting HSV image
  return hsvImage;
}

// Function to convert an RGB image to its Y'CbCr
function convertToYcbcr(img) {
  // Create a new image for Y'CbCr
  let ycbcrImg = createImage(img.width, img.height);

  // Load pixel data for both input and output images
  img.loadPixels();
  ycbcrImg.loadPixels();

  // Iterate over each pixel in the input RGB image
  for (let y = 0; y < img.height; y++) {
    for (let x = 0; x < img.width; x++) {
      // Calculate the flat pixel array index for the current pixel
      let index = (y * img.width + x) * 4;

      // Extract normalized RGB values from the input image
      let pixRed = img.pixels[index + 0] / 255;
      let pixGreen = img.pixels[index + 1] / 255;
      let pixBlue = img.pixels[index + 2] / 255;

      // Y' (luma) calculation using the specified formula
      let yPrime = 0.299 * pixRed + 0.587 * pixGreen + 0.114 * pixBlue;
      // ChomaBlue calculation
      let cb = -0.169 * pixRed - 0.331 * pixGreen + 0.5 * pixBlue;
      // ChomaRed calculation
      let cr = 0.5 * pixRed - 0.419 * pixGreen - 0.081 * pixBlue;

      // Shift and scale the Y'CbCr values to the [0, 255] range
      yPrime *= 255;
      cb = (cb + 0.5) * 255;
      cr = (cr + 0.5) * 255;

      // Assign Y'CbCr values to the corresponding pixels in the output image
      ycbcrImg.pixels[index + 0] = yPrime; // Y' component
      ycbcrImg.pixels[index + 1] = cb; // Cb component
      ycbcrImg.pixels[index + 2] = cr; // Cr component
      ycbcrImg.pixels[index + 3] = 255; // Alpha (fully opaque)
    }
  }

  // Update the pixel array for the Y'CbCr image
  ycbcrImg.updatePixels();

  // Return the resulting Y'CbCr image
  return ycbcrImg;
}

// Function to segment an image based on a specified threshold value
function segmentImage(img, threshold) {
  // Create a new image for the segmented result
  let segmentedImg = createImage(img.width, img.height);

  // Load pixel data for both input and output images
  img.loadPixels();
  segmentedImg.loadPixels();

  // Iterate over each pixel in the input image
  for (let i = 0; i < img.pixels.length; i += 4) {
    // Extract HSV values from the input image
    let hue = img.pixels[i + 0];
    let sat = img.pixels[i + 1];
    let val = img.pixels[i + 2];

    // Conditions based on the segmentation criteria
    if (val > threshold) {
      // Retain pixels above the threshold
      segmentedImg.pixels[i + 0] = hue;
      segmentedImg.pixels[i + 1] = sat;
      segmentedImg.pixels[i + 2] = val;
      segmentedImg.pixels[i + 3] = 255; // Fully opaque
    } else {
      // Set pixels below the threshold to black
      segmentedImg.pixels[i + 0] = 0;
      segmentedImg.pixels[i + 1] = 0;
      segmentedImg.pixels[i + 2] = 0;
      segmentedImg.pixels[i + 3] = 255; // Fully opaque
    }
  }

  // Update the pixel array for the segmented image
  segmentedImg.updatePixels();

  // Return the resulting segmented image
  return segmentedImg;
}

// Function to handle key presses (toggling the applyEffect variable)
function keyPressed() {
  // Toggle the applyEffect variable on and off when a key is pressed
  applyEffect = !applyEffect;
}

// Function to detect faces in an input image using the specified detector
function detectFaceFromImg(inputImage) {
  // Detect faces using the specified detector
  faces = detector.detect(inputImage.canvas);

  // Display the input image in a specific area of the canvas
  image(inputImage, 20, 580, 160, 120);

  // Iterate over each detected face
  for (let i = 0; i < faces.length; i++) {
    let face = faces[i];

    // Check if the face confidence is above a certain threshold
    if (face[4] > 4) {
      // Output the frame with face detection information
      outputFaceDetectionFrame();

      // Check if the applyEffect variable is true
      if (applyEffect) {
        // Choose an effect based on the key pressed
        switch (key) {
          case "g":
          case "G":
            // Apply grayscale filter
            grayScaleFilter();
            break;
          case "b":
          case "B":
            // Apply blur filter
            blurFilter();
            break;
          case "h":
          case "H":
            // Apply HSV filter
            HSVFilter();
            break;
          case "p":
          case "P":
            // Apply pixelated filter
            pixelatedFilter();
            break;
          case "y":
          case "Y":
            // Apply YCbCr filter
            YCbCrFilter();
            break;
          case "m":
          case "M":
            // Apply merged filter
            mergeFilter();
            break;
        }
      }
    }
  }
}

// Function to visually output the frame with face detection information
function outputFaceDetectionFrame() {
  // Store the current drawing state
  push();

  // Set stroke properties for the face rectangle
  strokeWeight(2);
  stroke(0, 0, 255);
  noFill();

  // Iterate through each detected face
  for (let i = 0; i < faces.length; i++) {
    let face = faces[i];

    // Check if the confidence level for the face is above the threshold
    if (face[4] > 4) {
      // Draw a rectangle around the detected face
      rect(face[0] + 20, face[1] + 580, face[2], face[3]);
    }
  }

  // Restore the previous drawing state
  pop();
}

// Function to apply a grayscale filter to the detected faces
function grayScaleFilter() {
  // Iterate through each detected face
  for (let i = 0; i < faces.length; i++) {
    let face = faces[i];

    // Check if the confidence level for the face is above the threshold
    if (face[4] > 4) {
      // Extract the position and dimensions of the detected face
      let imgX = face[0];
      let imgY = face[1];
      let imgW = face[2];
      let imgH = face[3];

      // Extract the detected face from the screenshot
      let detectedFace = screenshot.get(imgX, imgY, imgW, imgH);

      // Apply the grayscale filter to the detected face
      detectedFace = grayScale(detectedFace);

      // Display the grayscale face in the modified position
      image(detectedFace, imgX + 19, imgY + 580, imgW, imgH);
    }
  }
}

// Function to apply convolution operation
function convolution(x, y, matrix, img) {
  let matrixSize = matrix.length;
  let totalRed = 0.0;
  let totalGreen = 0.0;
  let totalBlue = 0.0;
  let offset = floor(matrixSize / 2);

  // Convolution matrix loop
  for (let i = 0; i < matrixSize; i++) {
    for (let j = 0; j < matrixSize; j++) {
      // Get pixel location within convolution matrix
      let xloc = x + i - offset;
      let yloc = y + j - offset;
      let index = (xloc + img.width * yloc) * 4;
      // Ensure we don't address a pixel that doesn't exist
      index = constrain(index, 0, img.pixels.length - 1);

      // Multiply all values with the mask and sum up
      totalRed += img.pixels[index + 0] * matrix[i][j];
      totalGreen += img.pixels[index + 1] * matrix[i][j];
      totalBlue += img.pixels[index + 2] * matrix[i][j];
    }
  }
  // Return the new color as an array
  return [totalRed, totalGreen, totalBlue];
}

// Function to apply convolution with a specified kernel to an image
function applyConvolution(img, kernel) {
  // Get the width and height of the input image
  let w = img.width;
  let h = img.height;

  // Create a new image to store the convolution result
  let result = createImage(w, h);

  // Load pixels for both the input image and the result image
  result.loadPixels();
  img.loadPixels();

  // Iterate over each pixel in the input image, excluding the borders
  for (let x = 1; x < w - 1; x++) {
    for (let y = 1; y < h - 1; y++) {
      // Apply the convolution operation to obtain new RGB values
      let [r, g, b] = convolution(x, y, kernel, img);

      // Calculate the pixel index in the result image
      let resultIndex = (y * w + x) * 4;

      // Assign the new RGB values to the result image pixels
      result.pixels[resultIndex] = r;
      result.pixels[resultIndex + 1] = g;
      result.pixels[resultIndex + 2] = b;
      result.pixels[resultIndex + 3] = 255; // Alpha channel (fully opaque)
    }
  }

  // Update the pixel array for the result image
  result.updatePixels();

  // Return the result image with applied convolution
  return result;
}

// Function to generate a Gaussian blur kernel
function getGaussianKernel(size, sigma) {
  // Initialize an empty array to store the kernel values
  let kernel = [];
  // Initialize a variable to store the sum of kernel values
  let sum = 0;

  // Iterate over rows of the kernel
  for (let i = 0; i < size; i++) {
    // Initialize an array for each row
    kernel[i] = [];
    // Iterate over columns of the kernel
    for (let j = 0; j < size; j++) {
      // Calculate the distance from the center of the kernel
      let x = i - Math.floor(size / 2);
      let y = j - Math.floor(size / 2);

      // Calculate the Gaussian function value and assign it to the kernel
      kernel[i][j] =
        Math.exp(-(x * x + y * y) / (2 * sigma * sigma)) /
        (2 * Math.PI * sigma * sigma);

      // Accumulate the value to calculate the sum
      sum += kernel[i][j];
    }
  }

  // Normalize the kernel by dividing each element by the sum
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      kernel[i][j] /= sum;
    }
  }

  // Return the resulting Gaussian blur kernel
  return kernel;
}

// Function to apply Gaussian blur to the image
function blurConversion(img) {
  let size = 10;
  let sigma = 8;
  let kernel = getGaussianKernel(size, sigma);
  return applyConvolution(img, kernel);
}

// Function to apply blur filter to detected faces
function blurFilter() {
  for (let i = 0; i < faces.length; i++) {
    let face = faces[i];
    if (face[4] > 4) {
      let imgX = face[0];
      let imgY = face[1];
      let imgW = face[2];
      let imgH = face[3];

      // Apply the Gaussian blur conversion to the detected face
      let detectedFace = screenshot.get(imgX, imgY, imgW, imgH);
      detectedFace = blurConversion(detectedFace);
      // Display the blurred face
      image(detectedFace, imgX + 19, imgY + 580, imgW, imgH);
    }
  }
}

// Function to apply HSV filter to detected faces
function HSVFilter() {
  // Iterate over the detected faces
  for (let i = 0; i < faces.length; i++) {
    let face = faces[i];
    // Check if confidence level for the face is above the threshold
    if (face[4] > 4) {
      // Extract position and dimensions of the detected face
      let imgX = face[0];
      let imgY = face[1];
      let imgW = face[2];
      let imgH = face[3];

      // Extract the detected face from the screenshot
      let detectedFace = screenshot.get(imgX, imgY, imgW, imgH);

      // Apply HSV filter to the detected face
      detectedFace = convertRGBToHSV(detectedFace);

      // Display the result of the HSV filter
      image(detectedFace, imgX + 19, imgY + 580, imgW, imgH);
    }
  }
}

// Function to pixelate the image using a specified block size
function convertToPixel(img, blockSize) {
  // Create a new image to store the pixelated result
  let outImage = createImage(img.width, img.height);
  outImage.loadPixels();

  // Iterate through the image with the specified block size
  for (let x = 0; x < img.width; x += blockSize) {
    for (let y = 0; y < img.height; y += blockSize) {
      // Calculate average pixel intensity for the block
      let sumR = 0,
        sumG = 0,
        sumB = 0;

      // Iterate through each pixel in the block
      for (let i = 0; i < blockSize; i++) {
        for (let j = 0; j < blockSize; j++) {
          // Get the pixel at the current position
          let px = img.get(x + i, y + j);
          // Accumulate color channel values
          sumR += red(px);
          sumG += green(px);
          sumB += blue(px);
        }
      }

      // Calculate average color for the block
      let avePixInt = color(
        sumR / (blockSize * blockSize),
        sumG / (blockSize * blockSize),
        sumB / (blockSize * blockSize)
      );

      // Paint the entire block using the average color
      for (let i = 0; i < blockSize; i++) {
        for (let j = 0; j < blockSize; j++) {
          outImage.set(x + i, y + j, avePixInt);
        }
      }
    }
  }

  // Update the pixel data for the output image
  outImage.updatePixels();
  // Return the pixelated image
  return outImage;
}

// Function to apply a pixelated filter to detected faces
function pixelatedFilter() {
  // Iterate through each detected face
  for (let i = 0; i < faces.length; i++) {
    let face = faces[i];
    // Check if confidence level for the face is above the threshold
    if (face[4] > 4) {
      let imgX = face[0];
      let imgY = face[1];
      let imgW = face[2];
      let imgH = face[3];

      // Extract the detected face from the screenshot
      let detectedFace = screenshot.get(imgX, imgY, imgW, imgH);

      // Apply grayscale filter to the detected face
      detectedFace = grayScale(detectedFace);
      // Apply pixelated filter to the grayscale face with a block size of 5 pixels
      detectedFace = convertToPixel(detectedFace, 5);

      // Display the pixelated face on the canvas
      image(detectedFace, imgX + 19, imgY + 580, imgW, imgH);
    }
  }
}

// Below are all my extension codes

// Function to apply Y'CbCr color space transformation to detected faces
function YCbCrFilter() {
  // Iterate through each detected face
  for (let i = 0; i < faces.length; i++) {
    let face = faces[i];
    // Check if confidence level for the face is above the threshold
    if (face[4] > 4) {
      let imgX = face[0];
      let imgY = face[1];
      let imgW = face[2];
      let imgH = face[3];

      // Extract the detected face from the screenshot
      let detectedFace = screenshot.get(imgX, imgY, imgW, imgH);

      // Apply Y'CbCr color space transformation to the detected face
      detectedFace = convertToYcbcr(detectedFace);

      // Display the face with Y'CbCr color space transformation on the canvas
      image(detectedFace, imgX + 19, imgY + 580, imgW, imgH);
    }
  }
}

// Function to merge multiple image processing effects on detected faces
function mergeFilter() {
  // Iterate through each detected face
  for (let i = 0; i < faces.length; i++) {
    let face = faces[i];
    // Check if confidence level for the face is above the threshold
    if (face[4] > 4) {
      let imgX = face[0];
      let imgY = face[1];
      let imgW = face[2];
      let imgH = face[3];

      // Extract the detected face from the screenshot
      let detectedFace = screenshot.get(imgX, imgY, imgW, imgH);

      // Apply a series of image processing effects to the detected face
      detectedFace = grayScale(detectedFace);
      detectedFace = convertToPixel(detectedFace, 5);
      detectedFace = blurConversion(detectedFace);

      // Display the face with merged image processing effects on the canvas
      image(detectedFace, imgX + 19, imgY + 580, imgW, imgH);
    }
  }
}

// Callback function triggered when the FaceAPI model is successfully loaded
function modelReady() {
  console.log("Model Loaded");

  // Initiate face detection once the model is ready
  faceapi.detect(gotResults);
}

// Callback function for processing face detection results
function gotResults(err, result) {
  if (err) {
    // Log any error that occurred during face detection
    console.error(err);
    return;
  }

  // Store the face detection results in the 'detections' variable
  detections = result;

  // If no screenshot is present, continue detecting faces
  if (!screenshot) {
    faceapi.detect(gotResults); // Keep detecting
  }
}

// Function to draw anime-style facial features based on face detection results
function drawAnimeFilter(detections) {
  // Iterate through each detected face
  detections.forEach((detection) => {
    // Extract information about the detected face
    const alignedRect = detection.alignedRect;
    const { _x, _y, _width, _height } = alignedRect._box;

    // Calculate the center, width, and height of the face for additional features like cat ears
    const faceCenter = { x: _x + _width / 2 + 200, y: _y + _height / 2 + 735 };
    const scaleFactor = getScaleFactor(_width); // Dynamic scaling based on face width

    // Prepare parameters for eyes drawing, including checking if they are blinking
    const leftEyeParts = detection.parts.leftEye;
    const rightEyeParts = detection.parts.rightEye;
    const eyes = getEyes(leftEyeParts, rightEyeParts);

    // Draw anime-style eyes with blinking detection
    drawAnimeEyes(eyes, leftEyeParts, rightEyeParts, scaleFactor); // Ensure this function checks for blinking

    // Draw a simplified nose
    const nose = detection.parts.nose;
    const noseBottom = nose[6]; // Approximate the bottom of the nose
    drawAnimeNose(noseBottom, scaleFactor);

    // Add cat ears and whiskers based on the face's position
    drawCatEars(faceCenter, _width, _height, scaleFactor);
    drawWhiskers(faceCenter, _width, _height, scaleFactor);
  });
}

function getScaleFactor(faceWidth) {
  // Define a base width as a reference for when the face is 'neutral' distance from the camera
  const baseWidth = 150; // Change this based on experimentation
  return faceWidth / baseWidth;
}

// Function to draw anime-style eyes based on detected eye parts
function drawAnimeEyes(eyes, leftEyeParts, rightEyeParts, scaleFactor) {
  // Determine if either eye is blinking
  const leftBlinking = isBlinking(leftEyeParts);
  const rightBlinking = isBlinking(rightEyeParts);

  // Define initial eye dimensions
  const eyeWidth = 50 * scaleFactor;
  const eyeHeight = leftBlinking || rightBlinking ? 10 : eyeWidth * 0.7; // Increase eye height

  // Calculate eye positions adjusted for the face
  let leftEyeX = eyes.leftEye.x + 200;
  let leftEyeY = eyes.leftEye.y + 735;
  let rightEyeX = eyes.rightEye.x + 200;
  let rightEyeY = eyes.rightEye.y + 735;

  // Define eye colors and shapes
  const eyeWhite = "#FFFFFF";
  const eyeColor = "#48220d";
  const pupilColor = "#000000";
  const highlightColor = "#ffffff";

  // Draw the white part of the eyes
  fill(eyeWhite);
  stroke(0);
  strokeWeight(3);
  ellipse(leftEyeX, leftEyeY, eyeWidth, eyeHeight);
  ellipse(rightEyeX, rightEyeY, eyeWidth, eyeHeight);

  // Draw the colored iris
  fill(eyeColor);
  noStroke();
  ellipse(leftEyeX, leftEyeY, eyeWidth * 0.6, eyeHeight * 0.8);
  ellipse(rightEyeX, rightEyeY, eyeWidth * 0.6, eyeHeight * 0.8);

  // Draw the vertical pupil
  fill(pupilColor);
  ellipse(leftEyeX, leftEyeY, eyeWidth * 0.2, eyeHeight * 0.5);
  ellipse(rightEyeX, rightEyeY, eyeWidth * 0.2, eyeHeight * 0.5);

  // Add highlight to the eye
  fill(highlightColor);
  ellipse(
    leftEyeX - eyeWidth * 0.15,
    leftEyeY - eyeHeight * 0.15,
    eyeWidth * 0.1,
    eyeHeight * 0.1
  );
  ellipse(
    rightEyeX - eyeWidth * 0.15,
    rightEyeY - eyeHeight * 0.15,
    eyeWidth * 0.1,
    eyeHeight * 0.1
  );

  const numLashes = 5; // Number of eyelashes per eye
  const lashLength = 10 * scaleFactor; // Length of eyelashes, adjusted by scaleFactor

  // Draw eyebrows for each eye
  drawEyebrow(leftEyeX, leftEyeY, eyeWidth, eyeHeight, scaleFactor);
  drawEyebrow(rightEyeX, rightEyeY, eyeWidth, eyeHeight, scaleFactor);

  // Draw eyelashes for each eye
  drawEyelashes(
    leftEyeX,
    leftEyeY,
    numLashes,
    lashLength,
    eyeWidth,
    eyeHeight,
    true
  );
  drawEyelashes(
    rightEyeX,
    rightEyeY,
    numLashes,
    lashLength,
    eyeWidth,
    eyeHeight,
    false
  );
}

// Function to draw eyelashes
function drawEyelashes(
  eyeX,
  eyeY,
  numLashes,
  lashLength,
  eyeWidth,
  eyeHeight,
  isLeftEye
) {
  stroke(0); // Eyelash color
  strokeWeight(2); // Eyelash thickness
  let angleIncrement = Math.PI / (numLashes + 1); // Divide the semicircle into equal parts based on the number of lashes
  let startAngle = isLeftEye ? Math.PI / 2 : -Math.PI / 2; // Start angle depending on the eye side

  for (let i = 1; i <= numLashes; i++) {
    let angle = startAngle + i * angleIncrement * (isLeftEye ? 1 : 1);
    let startX = eyeX + Math.cos(angle) * eyeWidth * 0.5;
    let startY = eyeY - Math.sin(angle) * eyeHeight * 0.5;
    let endX = startX + Math.cos(angle) * lashLength;
    let endY = startY - Math.sin(angle) * lashLength;
    line(startX, startY, endX, endY);
  }
}

// Draw eyebrows above the eyes
function drawEyebrow(eyeX, eyeY, eyeWidth, eyeHeight, scaleFactor) {
  noFill();
  stroke(0); // Eyebrow color
  strokeWeight(5 * scaleFactor); // Eyebrow thickness
  let eyebrowWidth = eyeWidth * 0.7;
  let eyebrowHeight = 10 * scaleFactor;
  let eyebrowY = eyeY - eyeHeight * 0.6; // Position the eyebrow above the eye
  beginShape();
  vertex(eyeX - eyebrowWidth / 2, eyebrowY);
  vertex(eyeX, eyebrowY - eyebrowHeight); // The peak of the eyebrow arch
  vertex(eyeX + eyebrowWidth / 2, eyebrowY);
  endShape();
}

function isBlinking(eyeParts) {
  // Calculate average distance between upper and lower eyelid points
  let distance = 10;
  for (let i = 0; i < eyeParts.length / 2; i++) {
    distance += dist(
      eyeParts[i]._x,
      eyeParts[i]._y,
      eyeParts[i + eyeParts.length / 2]._x,
      eyeParts[i + eyeParts.length / 2]._y
    );
  }
  distance /= eyeParts.length / 2;

  // Threshold for deciding if eye is closed
  return distance < 5;
}

function drawAnimeNose(noseBottom, scaleFactor) {
  fill(0);
  noStroke();
  // Simple nose just as a dot
  ellipse(
    noseBottom._x + 200,
    noseBottom._y + 735,
    5 * scaleFactor,
    5 * scaleFactor
  ); // Adjusted based on scaleFactor
}

function drawEar(x, y, earWidth, earHeight, rotation, scaleX) {
  const earColor = "#1f1f1f"; // Cat ear color
  const innerEarColor = "#FFC0CB"; // Lighter color for the inner ear
  const innerEarWidth = earWidth * 0.75; // Inner ear width
  const innerEarHeight = earHeight * 0.75; // Inner ear height

  push(); // Save the current transformation matrix
  translate(x, y); // Translate to the ear position
  rotate(radians(rotation)); // Rotate by specified degrees
  scale(scaleX, 1); // Flip the ear horizontally if needed

  // Outer ear
  fill(earColor);
  noStroke();
  triangle(0, 0, earWidth, 0, earWidth * 0.5, -earHeight);

  // Inner ear highlight
  fill(innerEarColor);
  triangle(
    earWidth * 0.25,
    0,
    earWidth * 0.75,
    0,
    earWidth * 0.5,
    -innerEarHeight
  );

  // Fur texture - simple lines to simulate fur at the edges
  stroke(0); // Black color for fur lines
  strokeWeight(2);
  line(earWidth * 0.3, -earHeight * 0.3, earWidth * 0.4, -earHeight * 0.4);
  line(earWidth * 0.6, -earHeight * 0.3, earWidth * 0.7, -earHeight * 0.4);

  pop(); // Restore the original transformation matrix
}

function drawCatEars(faceCenter, faceWidth, faceHeight, scaleFactor) {
  // Positions and sizes of the cat ears relative to the face
  const earWidth = faceWidth * 0.5 * scaleFactor;
  const earHeight = faceHeight * 0.75 * scaleFactor;
  const earXOffset = faceWidth * 0.65 * scaleFactor;

  // Left ear
  drawEar(
    faceCenter.x - earXOffset,
    faceCenter.y - faceHeight * 0.5,
    earWidth,
    earHeight,
    -25,
    1
  );

  // Right ear
  drawEar(
    faceCenter.x + earXOffset,
    faceCenter.y - faceHeight * 0.5,
    earWidth,
    earHeight,
    25,
    -1
  );
}

function drawWhiskers(faceCenter, faceWidth, faceHeight, scaleFactor) {
  stroke("#1f1f1f"); // Whisker colour
  strokeWeight(2);
  noFill();

  const whiskerLength = faceWidth * 0.7 * scaleFactor;
  const whiskerYOffset = faceHeight * 0.15 * scaleFactor;
  const whiskerXOffset = faceWidth * 0.25 * scaleFactor;

  // Left whiskers
  line(
    faceCenter.x - whiskerXOffset,
    faceCenter.y,
    faceCenter.x - whiskerXOffset - whiskerLength,
    faceCenter.y - whiskerYOffset
  );
  line(
    faceCenter.x - whiskerXOffset,
    faceCenter.y,
    faceCenter.x - whiskerXOffset - whiskerLength,
    faceCenter.y
  );
  line(
    faceCenter.x - whiskerXOffset,
    faceCenter.y,
    faceCenter.x - whiskerXOffset - whiskerLength,
    faceCenter.y + whiskerYOffset
  );

  // Right whiskers
  line(
    faceCenter.x + whiskerXOffset,
    faceCenter.y,
    faceCenter.x + whiskerXOffset + whiskerLength,
    faceCenter.y - whiskerYOffset
  );
  line(
    faceCenter.x + whiskerXOffset,
    faceCenter.y,
    faceCenter.x + whiskerXOffset + whiskerLength,
    faceCenter.y
  );
  line(
    faceCenter.x + whiskerXOffset,
    faceCenter.y,
    faceCenter.x + whiskerXOffset + whiskerLength,
    faceCenter.y + whiskerYOffset
  );
}

function getEyes(leftEyeParts, rightEyeParts) {
  // Calculate the center of the left eye
  let leftEyeCenter = getEyeCenter(leftEyeParts);

  // Calculate the center of the right eye
  let rightEyeCenter = getEyeCenter(rightEyeParts);

  return { leftEye: leftEyeCenter, rightEye: rightEyeCenter };
}

function getEyeCenter(eyeParts) {
  // Calculate the center of the eye based on the eye landmarks
  let sum = eyeParts.reduce(
    (acc, p) => {
      return { x: acc.x + p._x, y: acc.y + p._y };
    },
    { x: 0, y: 0 }
  );

  return { x: sum.x / eyeParts.length, y: sum.y / eyeParts.length };
}
