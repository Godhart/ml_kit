"""
Базовые функции по обработке текста
"""

import re


def purify(x):
    """
    Замена концов строк на пробелы, удаление символа с кодом 0xA0
    обрезка краевых пробелов, приведение к нижнему регистру
    """
    if isinstance(x, str):                # Если значение - строка:
        x = x.replace('\n', ' ').replace('\xa0', '').strip().lower()
    return x


def extract_year(value, pattern=r'\d\d.\d\d.(\d{4})', fallback_value=0):
    try:
        return int(re.search(pattern, value)[1])

    except (IndexError, TypeError, ValueError):
        return fallback_value
