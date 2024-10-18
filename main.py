import os

import pandas as pd
from dotenv import load_dotenv
import sys
sys.path.append('/home/choosen-one/choosen-one/EDU/hackathons/cont/hse-aiahp/app')


from app.models.yandexgptPRO import YandexGPTPRO
from app.models.deepseek import DeepSeek
from app.models.jailbreak import Jailbreak
from app.utils.submit import generate_submit

if __name__ == "__main__":
    load_dotenv()

    system_prompt = """
    Ты - профессиональный программист, учитель и ментор. Давай очень короткие ответы в одном предложении об ошибках.
    Я пришлю тебе условие, мое решение, описание ошибки и тесты, которые не проходят.
    Дай короткую подсказку в предложении, что нужно чтобы решить задачу.
    Объясни ошибку упомянув смысл задания.
    Не присылай код решения, присылай только короткую подсказку.
    Используй в речи следующие фразы: ваш код, выполняет условия, попробуйте изменить, условия задания, некорректно выполняет, скорректировать ошибку, или задавай наводящие вопросы, если нужно обратить внимание на ошибку в коде. Обращайся ко мне лично и на вы.
    """

    yandex_gpt = YandexGPTPRO(
        token=os.environ["YANDEX_GPT_IAM_TOKEN"],
        folder_id=os.environ["YANDEX_GPT_FOLDER_ID"],
        modelURI = os.environ["YANDEX_MODEL_URI"],
        system_prompt=system_prompt,
    )

    deep_gpt = DeepSeek(DeepSeek("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", 8192, 1, system_prompt))

    jailbreak = Jailbreak()


    def predict(row: pd.Series):
        return yandex_gpt.ask(deep_gpt.ask(jailbreak.clean_answer(row["student_solution"])))


    generate_submit(
        test_solutions_path="../data/raw/test/solutions.xlsx",
        predict_func=predict,
        save_path="../data/processed/submission.csv",
        use_tqdm=True,
    )
