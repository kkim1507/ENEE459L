# Lab 05 — Which removal did you actually get?

Pruning: structured versus unstructured sparsity, and measuring what actually shrunk.

It is not a compression lab and it is not an accuracy lab. Lab 05 measures no accuracy at all. There is no dataset in this tree, no training loop, no fine-tune and no accuracy curve, and every benchmark record you produce will carry the accuracy field with the value unknown and the reason beside it.

That is a deliberate choice and not an omission. This course's own rules call an invented number on a result a fabrication, and a curve produced on a machine you have never seen — in a session you were not in — looks exactly like evidence while being none. So the field stays, named and unknown. A report that silently omits a quantity and a report that names it as unmeasured are very different documents, and only the second can be read by somebody who was not in the room.

## The lab question

You pruned 90% of weights. What got smaller—parameter count, disk file, tensor shape, or nothing? Two methods called "pruning" produce identical sparsity reports while measuring completely different things. One removes weights (sparsity appears in every metric). One masks weights in-place (file and speed unchanged). Can you tell them apart from the numbers?

## What is this lab is built around

This lab is built around exposing the critical difference between setting neural network weights to zero and actually achieving hardware or memory savings through pruning. Without relying on external frameworks like PyTorch or NumPy, it contrasts **fine-grained (unstructured) pruning**—which merely masks values while leaving tensor shapes, parameter counts, and dense compute workloads unchanged—against **structured channel pruning**, which physically slices away contiguous output channels to yield genuinely smaller, faster dense tensors. Through rigorous disk byte accounting (`dense`, `masked`, and `sparse`) and structural taxonomy classifications (including 2:4 Ampere sparse patterns), the lab teaches students to evaluate pruning not by nominal request percentages, but by measurable reductions in storage footprint and hardware execution shape.


## What You Will Build

Six standalone probes reading sysfs, /proc, and CLI tools:

- **`magnitude_mask`**: Generates a binary mask of 0s and 1s marking the smallest-magnitude elements for unstructured removal based on a target pruning ratio.
- **`channel_keep`**: Identifies which output channel indices to retain by ranking channels by their $L^p$ norm while ensuring at least one channel survives.
- **`apply_mask`**: Sets masked positions in a tensor to zero without changing its overall shape, parameter count, or memory layout.
- **`drop_channels`**: Physically removes pruned output channels from axis 0 to return a genuinely smaller, contiguous dense tensor.Calculates the physical disk footprint in bytes for a set of tensors across dense, masked (framework/bitmap), or sparse formats.
- **`bytes_stored`**:Builds a standardized accounting record comparing requested nominal pruning ratios against actual achieved structural reductions, zero counts, disk bytes, and removal types.
- **`sparsity_row`**: Builds a standardized accounting record comparing requested nominal pruning ratios against actual achieved structural reductions, zero counts, disk bytes, and removal types.
- **`classify_removal`**: Categorizes the pruned state of tensors into a strict hardware-aware taxonomy (structurally absent, stored sparse, patterned 2:4, masked, or dense).
- **`sweep_model`**: Runs an end-to-end evaluation benchmark over an entire model across multiple pruning ratios and granularities to produce a full accounting table.

All functions use either a `Graph()` and/or `mock_macs` parameter for execution; see `main.py` as reference. When you cannot read something, you return `unknown(source, why)`, not a plausible default.

## How to Write code

The instructions for writing the code are provided in slides in `Module 1` on ELMS. The slide numbers for each of the function are mentioned below -

1. **magnitude_mask**: page 1
2. **channel_keep**: page 1
3. **apply_mask**: page 2.
4. **drop_channels**: page 2.
5. **bytes_stored**: page 2-3
6. **sparsity_row**: page 3
7. **classify_removal**: page 3-4
8. **sweep_model**: page 4

## How to clone lab05 code

```
git remote add upstream https://github.com/YOUR_USERNAME/YOUR_REPO.git
git pull upstream main
git push origin main
```

## How to Run

```bash
cd lab05/

# Generate the report on the board
python main.py
```

## How to compare responses

After completing the code, please validate the resulting JSON output file against `sample_complexity_results.json` to ensure it conforms to the expected format before submission. Use the following command to do so.

```bash
python compare_json.py ref_json.json your_json.json
```

## How to Debug your code

There are two ways to debug code - 
1. One way is to use breakpoints. For that, we use `pdb` the package and `pdb.set_trace()` to add a breakpoint at any line of code. 
2. Another way is to just the output by using command `print(out)` where out is output of any function. 

## How to save your work

For saving your work, you create a new branch named `solution4` by running the following command.
```bash
git switch -c solution5
```

After this, push your changes with following set of commands - 
```bash
git add .
git commit -m "Adding things"
git push -u origin solution5
```

## Before you hand in
Run your code and verify that its output matches the provided sample files. Once you have confirmed that everything is working correctly, push your changes to a new branch. Before submitting the link to your branch on Canvas, please verify that the branch contains the code you wrote and that all of your changes have been successfully pushed.
