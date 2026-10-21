# Lab 07 — What a frame costs before anything looks at it

A 1920×1080 frame in NV12 (ISP native) is 3.1 MB; in RGB (tutorial native) it is 6.2 MB. At 30 fps that is 178 MiB/second before inference starts. MobileNetV2 is 300M MACs; one frame read as 32-bit words is worth 311M MAC-equivalents at 200:1 energy ratio. The frame costs more than the network. Can you move the conversion from the CPU to the ISP's video codec, and can you measure what it buys?

## What is this lab is built around

Modern edge computer vision systems rarely fail because a neural network lacks capacity; they fail because the underlying hardware cannot move pixel data fast enough. This lab is built around the fundamental systems reality of edge computing: memory bandwidth is your primary bottleneck.

When streaming video from a physical camera sensor to an inference engine on a platform like the NVIDIA Jetson, frames undergo a continuous pipeline of physical and structural transformations. Raw sensor lines stream over CSI wires, hardware Image Signal Processors (ISPs) pack and demosaic pixel planes, software frameworks reorder color channels in system RAM, and neural net preprocessors resize and cast integer pixels into heavy 32-bit floating-point tensors.

Every single transformation incurs an explicit physical tax on the system bus. In this lab, you move beyond treating OpenCV and GStreamer as opaque black boxes. You will explore how pixels are physically packed across memory boundaries, why planar and semi-planar formats (NV12, I420) exist, and how to rigorously audit the memory transactions required to ingest live visual streams.


## What You Will Build

In this lab, you will build an end-to-end frame capture, memory budgeting, and diagnostic pipeline that interfaces directly with physical camera hardware. Specifically, you will implement:

- A Hardware Ingestion & Capture Engine: A robust, headless GStreamer pipeline using OpenCV that handles sensor warmup, synchronizes auto-exposure and auto-white-balance, and safely pulls frames without crashing the camera daemon or relying on an active GUI display server.

- A Memory Bandwidth Pricing Calculator: A mathematical accounting engine (frame_bytes, bytes_per_second, conversion_bytes, resize_bytes, and normalise_bytes) that prices the exact read/write memory transactions incurred by uncompressed camera formats, format conversions (e.g., NV12 to BGR), spatial downsampling, and float tensor normalization.

- An Automated Sensor Diagnostic Classifier: A statistical validation routine (frame_stats and classify) that calculates the primary moments of incoming pixel luma (min, max, mean, standard deviation, and dynamic spread) to reliably detect hardware failures—distinguishing dead black sensors and featureless flat fields from underexposed rooms and valid scenes.

- An Automated Artifact & Telemetry Generator: A structured test bench that captures a physical frame, writes the source image to disk, evaluates bus throughput, and exports a comprehensive, machine-readable JSON telemetry report profiling the full cost of the vision pipeline.



## How to Write code

The instructions for writing the code are provided in slides in `Module 1` on ELMS. The slide numbers for each of the function are mentioned below -

1. **frame_bytes**: pages 1
2. **bytes_per_second**: page 2-3
3. **frame_stats**: page 3-4
4. **classify**: page 4.
5. **conversion_bytes**: page 5
6. **resize_bytes**: page 5
7. **normalise_bytes**: page 6
8. **summarise_formats**: page 6-7

## How to clone lab07 code

```
git remote add upstream https://github.com/YOUR_USERNAME/YOUR_REPO.git
git pull upstream main
git push origin main
```

## How to Run

```bash
cd lab07/

# Generate the report on the board
python frames.py
```

## How to Debug your code

There is only one way to debug code - 
1. One way is to use breakpoints. For that, we use `pdb` the package and `pdb.set_trace()` to add a breakpoint at any line of code. 

## How to save your work

For saving your work, you create a new branch named `solution6` by running the following command.
```bash
git switch -c solution7
```

After this, push your changes with following set of commands - 
```bash
git add .
git commit -m "Adding things"
git push -u origin solution7
```

## Before you hand in
Run your code and verify that its output matches the provided sample files. Once you have confirmed that everything is working correctly, push your changes to a new branch. Before submitting the link to your branch on Canvas, please verify that the branch contains the code you wrote and that all of your changes have been successfully pushed.
