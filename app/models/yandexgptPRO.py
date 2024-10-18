import os
from typing import Optional

import requests
import sys
sys.path.append('/home/choosen-one/choosen-one/EDU/hackathons/cont/hse-aiahp')
from app.models.base import BaseModel


class YandexGPTPRO(BaseModel):
    """See more on https://yandex.cloud/en-ru/docs/foundation-models/concepts/yandexgpt/models"""

    model_urls = {        
        "pro": "gpt://{}/yandexgpt/latest"
    }

    def __init__(
        self,
        token: str,
        folder_id: str,
        modelURI: str,
        model_name: str = "pro",
        system_prompt: Optional[str] = None,        
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> None:
        super().__init__(system_prompt)
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "x-folder-id": folder_id,
        }
        self.model_url = YandexGPTPRO.model_urls[model_name].format(folder_id)
        self.completion_options = {
            "stream": False,
            "temperature": temperature,
            "maxTokens": str(max_tokens),
        }

    def ask(self, user_message: str, clear_history: bool = True) -> Optional[str]:
        if clear_history:
            self.messages = []
            if self.system_prompt:
                self.messages.append({"role": "system", "text": self.system_prompt})

        self.messages.append({"role": "user", "text": user_message})

        json_request = {
            "modelUri": self.model_url,  # Исправляем на self.model_url
            "completionOptions": self.completion_options,
            "messages": self.messages,
        }

        response = requests.post(self.api_url, headers=self.headers, json=json_request)
        if response.status_code != 200:
            print("Error:", response.status_code, response.text)
            return None

        response_data = response.json()
        assistant_message = response_data["result"]["alternatives"][0]["message"]["text"]
        self.messages.append({"role": "assistant", "text": assistant_message})
        return assistant_message


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    yandex_gpt = YandexGPTPRO(
        token=os.environ["YANDEX_GPT_IAM_TOKEN"],
        folder_id=os.environ["YANDEX_GPT_FOLDER_ID"],
        modelURI = os.environ["YANDEX_MODEL_URI"],
        system_prompt="Ты - профессиональный биолог. Отвечай коротко и по делу в научных терминах.",
    )
    print(yandex_gpt.ask("Кто такой манул?"))
