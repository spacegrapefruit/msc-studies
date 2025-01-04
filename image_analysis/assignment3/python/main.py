import logging
import pathlib
import typer

from image_pipelines import (
    circuit_board_qa_pipeline,
    filled_bottles_pipeline,
    fish_signal_counts_pipeline,
)


app = typer.Typer()

OUTPUT_DIR = pathlib.Path("data/output")


@app.command()
def fish_signal_counts(
    input_path_acridine: str,
    input_path_fits: str,
    input_path_dapi: str,
):
    output_dir = OUTPUT_DIR / "fish_signal_counts"
    output_dir.mkdir(parents=True, exist_ok=True)

    fish_signal_counts_pipeline(input_path_acridine, input_path_fits, input_path_dapi, output_dir)


@app.command()
def circuit_board_qa(input_path: str):
    output_dir = OUTPUT_DIR / "circuit_board_qa"
    output_dir.mkdir(parents=True, exist_ok=True)

    circuit_board_qa_pipeline(input_path, output_dir)


@app.command()
def filled_bottles(input_path: str):
    output_dir = OUTPUT_DIR / "filled_bottles"
    output_dir.mkdir(parents=True, exist_ok=True)

    improperly_filled = filled_bottles_pipeline(input_path, output_dir)
    print(f"Found {len(improperly_filled)} improperly filled bottles.")
    for i, bottle in enumerate(improperly_filled, start=1):
        print(f"  {i}. Bottle at column {bottle[0]}, height {bottle[1]}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    app()
