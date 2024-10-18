import sys
sys.path.append('/home/choosen-one/choosen-one/EDU/hackathons/cont/hse-aiahp')

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import Optional
from app.models.base import BaseModel

class DeepSeek(BaseModel):
    def __init__(self, model_name: str, max_model_len: int, tp_size: int, system_prompt: Optional[str] = None) -> None:
        super().__init__(system_prompt)
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.tp_size = tp_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(model=model_name, tensor_parallel_size=tp_size, max_model_len=max_model_len, trust_remote_code=True, enforce_eager=True)
        self.sampling_params = SamplingParams(temperature=0.3, max_tokens=256, stop_token_ids=[self.tokenizer.eos_token_id])

    def ask(self, user_message: str, clear_history: bool = True) -> Optional[str]:
        if clear_history:
            self.messages = []

        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

        self.messages.append({"role": "user", "content": user_message})

        prompt_token_ids = self.tokenizer.apply_chat_template(self.messages, add_generation_prompt=True)
        outputs = self.llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=self.sampling_params)

        result = outputs[0].outputs[0].text
        return result

    def generate_comments(self, solutions: pd.DataFrame) -> pd.DataFrame:
        author_comments = []

        for idx, row in solutions.iterrows():
            task_description = row['description']
            student_solution = row['student_solution']
            synt_errors = row['message'] if not pd.isna(row['message']) else "нет синтаксических ошибок"

            user_message = f'''
            Условие задачи:
            {task_description}

            Мое решение с ошибкой:
            {student_solution}

            Синтаксические ошибки в решении:
            {synt_errors}
            '''

            author_comment = self.ask(user_message, clear_history=True)
            author_comments.append({"id": row['id'], "author_comment": author_comment})

        submit_df = pd.DataFrame(author_comments)
        return submit_df


# Пример использования
if __name__ == "__main__":
    # Загрузка данных
    solutions = pd.read_excel('../../data/for_teams/test/solutions.xlsx')
    tasks = pd.read_excel('../../data/for_teams/test/tasks.xlsx')
    tests = pd.read_excel('../../data/for_teams/test/tests.xlsx')
    tests_with_errors = pd.read_csv('../../data/new_data/test/test_solutions_with_flake8.csv')

    solutions = tests_with_errors.merge(tasks[['id', 'description', 'author_solution']], left_on='task_id', right_on='id', how='left', suffixes=('', '_task'))

    # Инициализация модели
    model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
    max_model_len, tp_size = 8192, 1
    system_prompt = '''
    Ты - профессиональный программист и учитель.
    Я пришлю тебе условие задачи, верное решение и мое решение с ошибками.
    Дай крайне короткий ответ в одном предложении об ошибке в решении.
    Ошибки могут быть синтаксические и логические.
    Объясни ошибку через смысл задания. Ни в коем случае не ссылайся на реализацию в верном решении.
    Ни в коем случае не присылай исправленный код решения.
    Используй следующие фразы: "ваш код", "выполняет условия", "попробуйте изменить", "условия задания", "некорректно выполняет", "скорректировать ошибку" или задавай наводящие вопросы, если нужно обратить внимание на ошибку в коде.
    '''

    deep_seek = DeepSeek(model_name, max_model_len, tp_size, system_prompt)

    # Генерация комментариев
    submit_df = deep_seek.generate_comments(solutions)

    # Вывод результата
    print(submit_df[['id', 'author_comment']].head())