# Lab 02 — Is this the toolchain the course locked, and how do you know?

Lab 01 established what the machine is. This lab establishes what is installed on
it, which is a harder question than it sounds, because the software stack has a
failure mode the hardware does not: **it can be wrong and still work.**

A board with the wrong PyTorch wheel runs your code. It imports cleanly. It
prints a version number that looks right. It runs your model on the CPU at a
thirtieth of the speed and never says a word about it, and the first thing you
will notice is that your week-seven numbers are strange in a way you cannot
explain.

## The lab question

**Is every component on this unit the version the course pinned — and for the
ones you cannot confirm, do you know that you cannot confirm them?**

## The two machines this lab is built around

Both of these pass a casual inspection. Both have `pip list` output that looks
correct. They fail in different weeks, for different reasons, and telling them
apart is the entire assignment.

**The stock wheel.** `pip install torch` on an aarch64 machine succeeds. It
fetches a wheel from PyPI that was never built against CUDA. `import torch`
works, `torch.__version__` prints `2.5.0`, and `torch.cuda.is_available()`
returns `False`. The only trace in `pip list` is what is *missing*: an NVIDIA
build carries a local version tag like `2.5.0a0+872d972e41.nv24.08`, and a stock
one does not. Five characters at the end of a version string, which everybody
skims past.

**The sealed virtual environment.** TensorRT on Jetson is an apt package. There
is no pip wheel for it; it installs into the system `dist-packages`. A virtual
environment created without `--system-site-packages` therefore cannot see it —
and nothing complains, because nothing imports TensorRT until week nine, by
which point nobody remembers how the venv was made.

## What You Will Build

Five standalone probes reading sysfs, /proc, and CLI tools:

- **`probe_torch`**: Imports `PyTorch` to report its version, verify CUDA access and device name, and diagnose whether a stock PyPI wheel or environment misconfiguration is hiding the GPU..
- **`probe_cuda`**: Reads and parses `/usr/local/cuda/version.json` to verify that the CUDA toolkit manifest exists, extracts the installed CUDA version, and returns its major-minor release line..
- **`probe_opencv`**: Imports `cv2` to extract its version and calls `cv2.cuda.getCudaEnabledDeviceCount` to determine whether the library was built with GPU hardware acceleration or is the default CPU-only build.
- **`probe_tensorrt`**: Imports `tensorrt` to retrieve its installed version, flagging if an isolated virtual environment missing `--system-site-packages` is preventing access to system-level bindings.
- **`probe_l4t`**: Reads `/etc/nv_tegra_release` and uses regex to parse the board's Linux for Tegra (L4T) board support package release and revision numbers into a unified version string.

All reads use a `root` parameter so tests can inject fake filesystems. When you cannot read something, you return `unknown(source, why)`, not a plausible default.

## How to Write code

The instructions for writing the code are provided in slides in `Module 1` on ELMS. The slide numbers for each of the function are mentioned below -

1. **probe_torch**: page 4-5
2. **probe_cuda**: page 6
3. **probe_opencv**: page 7.
4. **probe_tensorrt**: page 8.
5. **probe_l4t**: page 9.

## How to clone lab02 code

```
git remote add upstream https://github.com/YOUR_USERNAME/YOUR_REPO.git
git pull upstream main
git push origin main
```

## How to Run

```bash
cd lab02/

# Generate the report on the board
python probes.py
```

After completing the code, please validate the resulting JSON output file against `sample_system_report.json` to ensure it conforms to the expected format before submission.

## How to Debug your code

There are two ways to debug code - 
1. One way is to use breakpoints. For that, we use `pdb` the package and `pdb.set_trace()` to add a breakpoint at any line of code. 
2. Another way is to just the output by using command `print(out)` where out is output of any function. 

## How to save your work

For saving your work, you create a new branch named `solution` by running the following command.
```bash
git switch -c solution2
```

After this, push your changes with following set of commands - 
```bash
git add .
git commit -m "Adding things"
git push -u origin solution2
```

## Analysis

1. **probe_torch**: probes if you have torch or not.
2. **probe_cuda**: probes if you have cuda or not.
3. **probe_opencv**: probes if you have opencv or not.
4. **probe_tensorrt**: probes if you have tensorrt or not.
5. **probe_l4t**: probes if you have l4t or not.

