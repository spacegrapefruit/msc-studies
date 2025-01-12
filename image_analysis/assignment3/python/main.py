import logging
import pathlib
import shutil
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
    input_path_fitc: str,
    input_path_dapi: str,
):
    """
    Counts the number of fish signals in a FISH image.
    """
    output_dir = OUTPUT_DIR / "fish_signal_counts"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = fish_signal_counts_pipeline(
        input_path_acridine=input_path_acridine,
        input_path_fitc=input_path_fitc,
        input_path_dapi=input_path_dapi,
        output_dir=output_dir,
        threshold_acridine=128,
        threshold_fitc=128,
        threshold_dapi=30,
    )

    logging.info("Cell analysis results")
    logging.info(f"Total cells: {len(results)}")
    logging.info("---------------------")

    for i, result in enumerate(results, start=1):
        logging.info(f"Cell {i}")
        logging.info(f"  X: {result['x']:.2f}, Y: {result['y']:.2f}")
        logging.info(f"  Area: {result['cell_area']}")
        logging.info(
            f"  Acridine={result['acridine_count']}, FITC={result['fitc_count']}, Ratio={result['acridine_to_fitc_ratio']}"
        )


@app.command()
def circuit_board_qa(input_path: str):
    """
    Detects defects in a circuit board image.
    """
    output_dir = OUTPUT_DIR / "circuit_board_qa"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = circuit_board_qa_pipeline(
        input_path=input_path,
        output_dir=output_dir,
    )

    logging.info("Circuit board QA results")
    logging.info(f"Total defects: {len(results)}")
    logging.info("---------------------")

    for i, result in enumerate(results, start=1):
        logging.info(f"Defect {i}")
        logging.info(f"  X: {result['x']:.2f}, Y: {result['y']:.2f}")
        logging.info(f"  Message: {result['message']}")


@app.command()
def filled_bottles(input_path: str):
    """
    Detects filled bottles in an image.
    """
    output_dir = OUTPUT_DIR / "filled_bottles"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = filled_bottles_pipeline(
        input_path=input_path,
        output_dir=output_dir,
    )

    logging.info("Bottle analysis results")
    logging.info(f"Total bottles: {len(results)}")
    logging.info("---------------------")

    for i, result in enumerate(results, start=1):
        logging.info(f"Bottle {i}")
        logging.info(f"  X: {result['x']:.2f}, Y: {result['y']:.2f}")
        logging.info(f"  Liquid level: {result['liquid_level']}")
        logging.info(
            f"  Neck-Shoulder range: {result['neck_level']}-{result['shoulder_level']}"
        )
        logging.info(f"  Filled: {result['is_filled']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    app()
