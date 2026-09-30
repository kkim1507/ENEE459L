# Lab 04 — Does model complexity predict latency?

Deploying deep neural networks onto resource-constrained edge hardware requires a precise understanding of both platform behavior and workload resource consumption. While high-level frameworks like PyTorch abstract away memory allocation and arithmetic scheduling, physical accelerators operate under strict, immovable boundaries dictated by available on-chip SRAM, system DRAM, and thermal dissipation limits. 

Performance analysis cannot rely on naive back-of-the-envelope formulas or the final size of a model file on disk; instead, engineers must evaluate exact tensor lifecycles and platform operational modes. This lab introduces you to the principles of static computational graph profiling, bridging the gap between abstract mathematical definitions of deep learning layers and their physical footprint on real-world compute engines.

## The lab question

How do an edge neural network’s architectural topologies, layer-level precisions, and execution dependencies dictate its true memory footprint and theoretical compute burden? 

Specifically, you will investigate how learnable parameters, persistent operational buffers, and dynamically co-existing activation tensors consume system capacity during an inference pass. By dissecting non-trivial structures, such as grouped convolutions that dramatically compress parameter volume and residual skip connections that prolong tensor residency, you will answer whether a candidate model can realistically satisfy an embedded platform’s memory budget before compiling a single kernel or running an inference forward pass.

## What is this lab is built around

This lab is built around the construction of a deterministic, framework-agnostic static graph analyzer that models neural network layers directly from their structural representations. Working with abstract layer definitions rather than heavyweight runtime weights, you will construct core telemetry functions: 

calculating learnable weight volumes across grouped convolutions and linear layers, auditing multi-precision tensor footprints that separate trainable weights from floating-point running statistics, and simulating compiler-style tensor liveness passes to determine exact peak resident activation memory. 

Finally, the lab grounds these static structural analyses by standardizing arithmetic metrics through formalized Multiply-Accumulate (MAC) to Floating-Point Operation (FLOP) conversions, giving you a rigorous, reproducible toolchain for evaluating edge deep learning models.


## What You Will Build

Six standalone probes reading sysfs, /proc, and CLI tools:

- **`count_parameters`**: Counts the trainable weights and biases across all layers, accounting for grouped convolutions and excluding non-learnable batch normalization buffers.
- **`model_size_bytes`**: Computes the total memory required to store the model by multiplying parameter and persistent buffer counts by their individual datatype byte widths.
- **`count_activations`**: Performs a forward liveness pass tracking tensor lifetimes across layer dependencies to calculate both total generated activation memory and true peak resident memory.
- **`to_flops`**: Converts an operation count from Multiply-Accumulates (MACs) to Floating-Point Operations (FLOPs) according to a specified, validated scaling convention.

All functions use either a `Graph()` and/or `mock_macs` parameter for execution; see `main.py` as reference. When you cannot read something, you return `unknown(source, why)`, not a plausible default.

## How to Write code

The instructions for writing the code are provided in slides in `Module 1` on ELMS. The slide numbers for each of the function are mentioned below -

1. **count_parameters**: pages 1,2
2. **model_size_bytes**: page 3
3. **count_activations**: page 4.
4. **to_flops**: page 5.

## How to clone lab04 code

```
git remote add upstream https://github.com/YOUR_USERNAME/YOUR_REPO.git
git pull upstream main
git push origin main
```

## How to Run

```bash
cd lab04/

# Generate the report on the board
python main.py
```

After completing the code, please validate the resulting JSON output file against `sample_complexity_results.json` to ensure it conforms to the expected format before submission.

## How to Debug your code

There are two ways to debug code - 
1. One way is to use breakpoints. For that, we use `pdb` the package and `pdb.set_trace()` to add a breakpoint at any line of code. 
2. Another way is to just the output by using command `print(out)` where out is output of any function. 

## How to save your work

For saving your work, you create a new branch named `solution4` by running the following command.
```bash
git switch -c solution4
```

After this, push your changes with following set of commands - 
```bash
git add .
git commit -m "Adding things"
git push -u origin solution4
```

## Before you hand in
Run your code and verify that its output matches the provided sample files. Once you have confirmed that everything is working correctly, push your changes to a new branch. Before submitting the link to your branch on Canvas, please verify that the branch contains the code you wrote and that all of your changes have been successfully pushed.
