import click
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity

TEST_SIZE = 325
EMBEDDING_SIZE = 768


def string2embedding(string: str) -> torch.Tensor:
    return torch.Tensor([float(i) for i in string.split()])


def embedding2string(embedding: torch.Tensor) -> str:
    return " ".join([str(i) for i in embedding.tolist()])


def _get_cosine_similarity(pred_df: pd.DataFrame, true_df: pd.DataFrame) -> float:
    predictions = pred_df["author_comment_embedding"]
    true_values = true_df["author_comment_embedding"]
    total_cos_sim = 0

    for idx in range(len(true_values)):
        pred_value = string2embedding(predictions.iloc[idx])
        gt_value = string2embedding(true_values.iloc[idx])

        if len(pred_value) != len(gt_value):
            raise ValueError(f"Embeddings have different sizes: {len(pred_value)} != {len(gt_value)}")

        cos_sim_value = cosine_similarity(pred_value.unsqueeze(0), gt_value.unsqueeze(0))
        total_cos_sim += cos_sim_value
    return float(total_cos_sim / len(true_df))


def _generate_random_dataframe() -> pd.DataFrame:
    dataframe = pd.DataFrame(columns=["solution_id", "author_comment", "author_comment_embedding"])
    for i in range(TEST_SIZE):
        random_embedding = torch.randn(EMBEDDING_SIZE)
        dataframe.loc[len(dataframe)] = [i, f"comment_{i}", random_embedding]
    return dataframe


def calculate_team_score(submit_path: str, gt_path: str) -> float:
    submit_df = pd.read_csv(submit_path)
    true_df = pd.read_excel(gt_path)
    submit_df = submit_df[submit_df["solution_id"].isin(true_df["id"])]
    return (_get_cosine_similarity(submit_df, true_df) - 0.6) / 0.4


def calculate_team_score_and_save(submit_path: str, gt_path: str, save_path: str) -> float:
    score = calculate_team_score(submit_path, gt_path)
    with open(save_path, "w") as f:
        f.write(f"{score}")
    return score


@click.command()
@click.argument("submit_path", type=click.Path(exists=True))
@click.argument("gt_path", type=click.Path(exists=True))
@click.argument("save_path", type=click.Path())
def main(submit_path, gt_path, save_path):
    """
    Calculate team score and save it to the file.

    SUBMIT_PATH: Path to the submission CSV file.
    GT_PATH: Path to the ground truth Excel file.
    SAVE_PATH: Path to save the resulting score.
    """
    score = calculate_team_score_and_save(submit_path, gt_path, save_path)
    click.echo(f"Score calculated and saved: {score}")


if __name__ == "__main__":
    main()
