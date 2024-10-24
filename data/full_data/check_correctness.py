import click
import pandas as pd

from score import TEST_SIZE, EMBEDDING_SIZE, string2embedding


def _is_ids_correct(submit_df: pd.DataFrame, submit_example_df: pd.DataFrame) -> bool:
    not_presented = set(submit_example_df["solution_id"]) - set(submit_df["solution_id"])
    not_needed = set(submit_df["solution_id"]) - set(submit_example_df["solution_id"])

    not_presented = list(not_presented)
    not_presented.sort()
    not_needed = list(not_needed)
    not_needed.sort()

    error_message = "Submit is incorrect."
    if len(not_presented) + len(not_needed) > 0:
        if len(not_presented) > 0:
            error_message += f" Not presented solution_id: {not_presented}."
        if len(not_needed) > 0:
            error_message += f" Not needed solution_id: {not_needed}."
        raise ValueError(error_message)
    return True


def _are_rows_match_size(submit_df: pd.DataFrame) -> bool:
    incorrect_rows = []
    for idx in range(TEST_SIZE):
        if len(string2embedding(submit_df["author_comment_embedding"].iloc[idx])) != EMBEDDING_SIZE:
            incorrect_rows.append(idx)
    if len(incorrect_rows) > 0:
        raise ValueError(f"Submit has incorrect rows: {incorrect_rows}. (incorrect size of embedding)")
    return True


def is_correct_submit(submit_path: str, submit_example_path: str) -> bool:
    if not submit_path.endswith(".csv"):
        raise ValueError(f"{submit_path} is not a .csv file.")

    submit_df = pd.read_csv(submit_path)
    submit_example_df = pd.read_csv(submit_example_path)

    _is_ids_correct(submit_df, submit_example_df)
    _are_rows_match_size(submit_df)

    return True


@click.command()
@click.argument("submit_path", type=click.Path(exists=True))
@click.argument("submit_example_path", type=click.Path(exists=True))
def main(submit_path, submit_example_path):
    """
    Validate the submission file.

    SUBMIT_PATH: Path to the submission CSV file.
    SUBMIT_EXAMPLE_PATH: Path to the example CSV file.
    """
    try:
        if is_correct_submit(submit_path, submit_example_path):
            click.echo("The submission file is correct.")
    except ValueError as e:
        click.echo(f"Validation failed: {e}")


if __name__ == "__main__":
    main()
