# Lab 06 — Eight bits, and the range you chose to spend them on.

Two INT8 quantized files are identical in size, both 12,460 bytes, both per-channel symmetric. Only one metric differs: which activation samples the calibrator was allowed to see. One flattens half the activation range onto a single quantized code. Which one did you ship, and can you tell from the file?

## What is this lab is built around

This lab is built around post-training quantization and the practical trade-offs between model compression and numerical accuracy. Using a deterministic convolutional neural network (`tinyconv`), it systematically investigates how moving from 32-bit floating-point weights and activations down to FP16 or INT8 formats affects disk storage and error rates. Central to the assignment is exploring the nuances of 8-bit quantization—contrasting whole-tensor versus per-channel scaling to handle channel-dependent weight distributions, evaluating symmetric versus asymmetric schemes on one-sided ReLU activations, and demonstrating how biased calibration sets (such as single-class activation data) cause severe clipping and degradation. Finally, it uses a multi-objective Pareto frontier analysis to identify the winning configurations that achieve the best balance between minimal effective bits per parameter and low reconstruction error.


## What You Will Build

Six standalone probes reading sysfs, /proc, and CLI tools:

- **quantize_tensor**: Converts floating-point values into bounded integers using calculated scaling factors and zero-point offsets.
- **dequantize**: Reconstructs approximate floating-point values from stored integer codes using their respective scales and zero-points.
- **quantization_error**: Measures precision loss between original and dequantized tensors by calculating Frobenius, relative, and worst-case absolute differences.
- **bytes_stored**: Calculates the total disk space required to store model weights along with any scale and zero-point metadata.
- **effective_bits**: Computes the average bit cost per weight parameter after accounting for metadata overhead.
- **precision_row**: Runs a quantization policy across weights and activations to compile an evaluation record of error, size, and clipping stats.
- **pareto_front**: Identifies optimal configurations by filtering out setups that are strictly outperformed across all target metrics.



## How to Write code

The instructions for writing the code are provided in slides in `Module 1` on ELMS. The slide numbers for each of the function are mentioned below -

1. **quantize_tensor**: pages 1-2
2. **dequantize**: page 2
3. **quantization_error**: page 3
4. **bytes_stored**: page 3.
5. **effective_bits**: page 4
6. **precision_row**: page 4-5
7. **pareto_front**: page 5-6

## How to clone lab06 code

```
git remote add upstream https://github.com/YOUR_USERNAME/YOUR_REPO.git
git pull upstream main
git push origin main
```

## How to Run

```bash
cd lab06/

# Generate the report on the board
python main.py
```

## How to Debug your code

There is only one way to debug code - 
1. One way is to use breakpoints. For that, we use `pdb` the package and `pdb.set_trace()` to add a breakpoint at any line of code. 

## How to save your work

For saving your work, you create a new branch named `solution6` by running the following command.
```bash
git switch -c solution6
```

After this, push your changes with following set of commands - 
```bash
git add .
git commit -m "Adding things"
git push -u origin solution6
```

## Before you hand in
Run your code and verify that its output matches the provided sample files. Once you have confirmed that everything is working correctly, push your changes to a new branch. Before submitting the link to your branch on Canvas, please verify that the branch contains the code you wrote and that all of your changes have been successfully pushed.
