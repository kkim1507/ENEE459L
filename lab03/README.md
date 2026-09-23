# Lab 03 — How long does one inference take, and why is that question ambiguous?

Lab 01 established what the machine is. Lab 02 established what is installed on it. This lab produces the first number, and the number is the easy part.

Ask ten people to measure one inference and you will get ten answers spread over a factor of forty. Not because anyone made a mistake — because "how long does one inference take" is six questions wearing one coat, and everybody silently answers a different one.

Different hardware takes different amounts of time to warm up. For example, running this model on the GPU takes 22 runs to settle because the GPU has to choose and optimize its kernels. Running the exact same model on the CPU settles in just 10 runs because it does not have that extra setup step.

If you hard-code samples[20:] to throw away the first 20 measurements:
- It works on the GPU.
- It throws away 10 perfectly good measurements on the CPU.
- It will fail completely on any slower machine that needs more than 20 runs to settle, leaving slow warm-up runs mixed into your real data.
Because warm-up time depends on the hardware, do not hard-code this number. Instead, write a function that detects when the run has settled, and log how many runs were discarded.

If a run never stabilizes (it keeps drifting), your function should discard nothing because there is no steady baseline to compare against. It is then is_stationary's job to flag that the benchmark drifted.

## The lab question

Which iteration? The first one on the GPU is around twenty-five times slower than the hundredth, and that is not noise. cuDNN benchmarks several convolution algorithms for your layer shape on first sight of it and keeps the fastest; after that the clock governor takes another twenty iterations to notice there is sustained work to do. Include those and you report a number no steady-state system will ever see. Exclude them and you report a number no cold-started system will ever see. Both are real; only one of them is the one you meant.

Where does the stopwatch start and stop? CUDA kernel launches are asynchronous. model(x) returns as soon as the work is queued, not when it is done. A loop that reads the clock on either side of it measures the launch, and the launch takes tens of microseconds regardless of how much work it queued. This produces impossible throughput — twenty-six thousand frames a second from a board that does one hundred and sixty — and it is the single most common way published edge-AI numbers are wrong.

## What is this lab is built around

You will be implementing a rigorous hardware benchmarking and statistical telemetry pipeline designed to collect reproducible performance profiles on NVIDIA Jetson systems. At its core, the timing workflow synchronizes asynchronous hardware tasks before and after every repetition, converting raw timestamps into clean, millisecond-level execution latencies. To ensure clean measurements, the pipeline includes diagnostic statistical functions: it detects and trims leading warm-up iterations based on the median run rate of the settled tail, compiles non-parametric summary statistics (mean, standard deviation, and linearly interpolated 50th, 95th, and 99th percentiles), and inspects sample spacing for multimodal splits that would indicate competing background processes or dynamic throttling.


## What You Will Build

Six standalone probes reading sysfs, /proc, and CLI tools:

- **`run_timed_iterations`**: Runs the workload repeatedly with hardware synchronization barriers before and after each pass to collect execution durations in milliseconds.
- **`find_warmup_boundary`**: Identifies and discards the initial sequence of iterations that exceed the settled tail's median by more than 50% due to cache and compilation warming.
- **`summarize`**: Calculates distribution statistics for the sample set, including the mean, sample standard deviation, extrema, and linearly interpolated 50th, 95th, and 99th percentiles.
- **`is_multimodal`**: Detects whether trimmed latency samples split into two distinct clusters across an unusually large gap, signaling competing processes or dynamic throttling.
- **`probe_power_state`**: Queries the active `nvpmodel` power profile and compares CPU minimum/maximum scaling limits to verify if clocks are pinned to fixed ceilings.
- **`probe_telemetry`**: Interrogates Linux `sysfs` paths to capture the peak SoC temperature across thermal zones, instantaneous board power from INA3221 monitor rails, and GPU load.

All functions use either a `Bench.real()` and/or `samples` parameter for execution. When you cannot read something, you return `unknown(source, why)`, not a plausible default.

## How to Write code

The instructions for writing the code are provided in slides in `Module 1` on ELMS. The slide numbers for each of the function are mentioned below -

1. **run_timed_iterations**: page 1
2. **find_warmup_boundary**: page 1
3. **summarize**: page 2.
4. **is_multimodal**: page 2.
5. **probe_power_state**: page 3.
5. **probe_telemetry**: page 3.

## How to clone lab03 code

```
git remote add upstream https://github.com/YOUR_USERNAME/YOUR_REPO.git
git pull upstream main
git push origin main
```

## How to Run

```bash
cd lab03/

# Generate the report on the board
python measure.py
```

After completing the code, please validate the resulting JSON output file against `sample_system_report.json` and `sample_samples_analysis.json` to ensure it conforms to the expected format before submission.

## How to Debug your code

There are two ways to debug code - 
1. One way is to use breakpoints. For that, we use `pdb` the package and `pdb.set_trace()` to add a breakpoint at any line of code. 
2. Another way is to just the output by using command `print(out)` where out is output of any function. 

## How to save your work

For saving your work, you create a new branch named `solution3` by running the following command.
```bash
git switch -c solution3
```

After this, push your changes with following set of commands - 
```bash
git add .
git commit -m "Adding things"
git push -u origin solution3
```

## Before you hand in
Run your code and verify that its output matches the provided sample files. Once you have confirmed that everything is working correctly, push your changes to a new branch. Before submitting the link to your branch on Canvas, please verify that the branch contains the code you wrote and that all of your changes have been successfully pushed.
