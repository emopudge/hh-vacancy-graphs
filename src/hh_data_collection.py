"""
Сбор данных о вакансиях с hh.ru
"""

from typing import List, Tuple
import pandas as pd
from session import Session
import os


def get_vacancies(pages: int, tag: str) -> List[Tuple[str, str, str, str]]:
    """
    Функция собирает уникальные вакансии по ключевому слову с сайта hh.ru

    :param pages: количество страниц для обработки
    :param tag: ключевое слово для поиска вакансий
    :return: список кортежей в формате (id, name, requirement, responsibility)
    """
    base_url = "https://api.hh.ru"
    unique_vacancies = {}

    session = Session(base_url=base_url)
    for page in range(pages):
        try:
            response = session.get("/vacancies", params={"text": tag, "page": page})
            data = response.json()

            for item in data.get("items", []):
                vacancy_id = item["id"]
                if vacancy_id not in unique_vacancies:
                    snippet = item.get("snippet", {})
                    requirement = snippet.get("requirement", "") or ""
                    responsibility = snippet.get("responsibility", "") or ""

                    # Чистим поля от HTML-тегов
                    requirement = requirement.replace("<highlighttext>", "").replace("</highlighttext>", "")
                    responsibility = responsibility.replace("<highlighttext>", "").replace("</highlighttext>", "")

                    unique_vacancies[vacancy_id] = (
                        vacancy_id,
                        item.get("name", "") or "",
                        (requirement or "").strip(),
                        (responsibility or "").strip(),
                    )

        except Exception as e:
            print(f"Ошибка при запросе к странице {page}: {e}")

    return list(unique_vacancies.values())


if __name__ == "__main__":
    vacancies = get_vacancies(3, "python")
    df = pd.DataFrame(vacancies, columns=["id", "title", "requirement", "responsibility"])
    df.to_csv("../data/python_300_vac.csv", index=False)
