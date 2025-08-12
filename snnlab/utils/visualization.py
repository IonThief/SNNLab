def draw_model_graph(model, input_size, fname=None, graph_depth=2):
    import os

    from torchview import draw_graph

    OUT_DIR = ".RENDERED_MODELS_AS_GRAPH"
    os.makedirs(OUT_DIR, exist_ok=True)

    fname = fname or model.__class__.__name__

    draw_graph(
        model,
        input_size=input_size,
        depth=graph_depth,
        expand_nested=True,
        graph_dir="TB",
    ).visual_graph.render(
        filename=fname,
        directory=OUT_DIR,
        format="svg",
        cleanup=True,
    )
    print(f"Model {model.__class__.__name__} graph saved to {OUT_DIR}/{fname}.svg")
