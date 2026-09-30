import json
from pathlib import Path
from graph import Graph, Layer, computed
from complexity import count_parameters, model_size_bytes, count_activations, to_flops

# create a dummy ResNet model
def build_synthetic_graph() -> Graph:
    """Builds a small ResNet-like toy network to exercise all paths:
    - Conv with bias=False and kernel=(3, 3)
    - Batch Norm (parametric + buffers)
    - Skip-connection (Layer.reads across layers)
    - Linear layer with bias
    """
    layers = [
        # 1. Stem conv
        Layer(
            name="conv1",
            kind="conv",
            in_shape=(3, 32, 32),
            out_shape=(16, 32, 32),
            kernel=(3, 3),
            bias=False,
            weight_dtype="fp32",
            act_dtype="fp32",
        ),
        # 2. Batch norm
        Layer(
            name="bn1",
            kind="bn",
            in_shape=(16, 32, 32),
            out_shape=(16, 32, 32),
            weight_dtype="fp32",
            act_dtype="fp32",
        ),
        # 3. Activation (zero params)
        Layer(
            name="relu1",
            kind="relu",
            in_shape=(16, 32, 32),
            out_shape=(16, 32, 32),
        ),
        # 4. Residual branch 1 (conv)
        Layer(
            name="res_conv",
            kind="conv",
            in_shape=(16, 32, 32),
            out_shape=(16, 32, 32),
            kernel=(3, 3),
            bias=False,
            reads=("relu1",),
        ),
        # 5. Skip connection add (combines relu1 and res_conv)
        Layer(
            name="add1",
            kind="add",
            in_shape=(16, 32, 32),
            out_shape=(16, 32, 32),
            reads=("relu1", "res_conv"),
        ),
        # 6. Global average pool
        Layer(
            name="pool",
            kind="pool",
            in_shape=(16, 32, 32),
            out_shape=(16, 1, 1),
        ),
        # 7. Flatten (spatial to 1D)
        Layer(
            name="flatten",
            kind="flatten",
            in_shape=(16, 1, 1),
            out_shape=(16,),
        ),
        # 8. Classification head
        Layer(
            name="fc",
            kind="linear",
            in_shape=(16,),
            out_shape=(10,),
            bias=True,
            weight_dtype="fp32",
            act_dtype="fp32",
        ),
    ]

    return Graph(
        name="toy_resnet",
        input_shape=(3, 32, 32),
        layers=layers,
        precision="fp32",
        source="synthetic_test",
    )

def main(output_path: str = "complexity_results.json") -> None:
    # 1. Setup graph (use synthetic, or load_graph("model.json") if available)
    graph = build_synthetic_graph()

    # Optional: validate structure first
    validation_issues = graph.validate()
    if validation_issues:
        print(f"Graph validation warnings: {validation_issues}")

    # 2. Run complexity functions
    params = count_parameters(graph)
    size = model_size_bytes(graph)
    activations = count_activations(graph)

    # 3. Test to_flops()
    # Provide a mock MAC finding to test both conversion conventions
    mock_macs = computed(
        value=1_500_000,
        source="mock_layer_pass",
        per_layer={"conv1": 432_000, "res_conv": 1_068_000},
    )
    flops_two = to_flops(mock_macs, convention="mac_is_two_flops")
    flops_one = to_flops(mock_macs, convention="mac_is_one_flop")

    # 4. Consolidate results
    results = {
        "model_name": graph.name,
        "layer_count": len(graph),
        "count_parameters": params,
        "model_size_bytes": size,
        "count_activations": activations,
        "to_flops": {
            "mac_is_two_flops": flops_two,
            "mac_is_one_flop": flops_one,
        },
    }

    # 5. Write to JSON
    dest = Path(output_path)
    dest.write_text(json.dumps(results, indent=2))
    print(f"Results successfully saved to {dest.resolve()}")

def debug_single_function(func):
    # 1. Setup graph (use synthetic, or load_graph("model.json") if available)
    graph = build_synthetic_graph()

    # Optional: validate structure first
    validation_issues = graph.validate()
    if validation_issues:
        print(f"Graph validation warnings: {validation_issues}")

    # if func is count_parameters, model_size_bytes, or count_activations, call it and print the result
    # out = func(graph)

    # if func is to_flops, call it with a mock MAC finding and print the result
    mock_macs = computed(
        value=1_500_000,
        source="mock_layer_pass",
        per_layer={"conv1": 432_000, "res_conv": 1_068_000},
    )
    flops_two = to_flops(mock_macs, convention="mac_is_two_flops")
    flops_one = to_flops(mock_macs, convention="mac_is_one_flop")

    # see the output with print below
    print()


if __name__ == "__main__":
    main()