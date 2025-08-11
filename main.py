import onnxruntime as ort
import typer

app = typer.Typer()


@app.command()
def run(
    model_path: str = typer.Argument(..., help="Path to the ONNX model file"),
    providers: list[str] = typer.Option(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "Execution providers to use, in order of preference",
    ),
):
    """
    Run inference on an ONNX model using the specified execution providers.
    """
    # Create the ONNX Runtime session
    sess = ort.InferenceSession(model_path, providers=providers)

    typer.echo(f"Loaded model from {model_path}")
    typer.echo(f"Using providers: {sess.get_providers()}")

    # TODO: Insert code here to prepare inputs and run session.run()


if __name__ == "__main__":
    app()
