"""Read Arabic numerals as words before sending text to ZipVoice.

ZipVoice's tokenizer accepts digit characters, but the acoustic model often
skips them.  Normalizing to Chinese/English number words also makes the
spoken result much more natural.

Rules:
  * ordinary quantities: 50 -> 五十 / fifty
  * decimals: 3.14 -> 三点一四 / three point one four
  * percentages: 50% -> 百分之五十 / fifty percent
  * long unbroken digit strings (phone/order/ID, 10+ digits) and numbers
    with leading zeroes are read digit-by-digit;
  * the same long number followed by a measure word/unit is read as a
    cardinal quantity.
"""
from __future__ import annotations

import re

ZH_DIGITS = "零一二三四五六七八九"
ZH_DIGIT_NAMES = {str(i): ZH_DIGITS[i] for i in range(10)}
ZH_UNITS = ("", "十", "百", "千")
ZH_GROUP_UNITS = ("", "万", "亿", "兆")

EN_ONES = ("zero", "one", "two", "three", "four", "five", "six",
           "seven", "eight", "nine", "ten", "eleven", "twelve",
           "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
           "eighteen", "nineteen")
EN_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty",
           "seventy", "eighty", "ninety")
EN_GROUP_UNITS = ("", "thousand", "million", "billion", "trillion",
                  "quadrillion")

# Characters/words immediately after a number that make it a quantity.
ZH_MEASURE = set("元块个只条件次岁人位名米克吨斤公里里步倍分秒年月日层间套台部页本张条")
EN_MEASURE = {
    "dollar", "dollars", "yuan", "piece", "pieces", "person", "persons",
    "people", "meter", "meters", "metre", "metres", "kilometer",
    "kilometers", "kilometre", "kilometres", "kg", "km", "year", "years",
    "month", "months", "day", "days", "hour", "hours", "minute",
    "minutes", "second", "seconds", "time", "times", "item", "items",
    "ton", "tons", "gram", "grams", "percent",
}


def _zh_group(n: int) -> str:
    """0..9999 -> Chinese cardinal (without group unit)."""
    if n <= 0:
        return ""
    out: list[str] = []
    zero = False
    for pos in range(3, -1, -1):
        d = (n // (10 ** pos)) % 10
        if d:
            if zero and out:
                out.append("零")
            # 10..19 use 十, not 一十
            if not (d == 1 and pos == 1 and not out):
                out.append(ZH_DIGITS[d])
            out.append(ZH_UNITS[pos])
            zero = False
        elif out:
            zero = True
    return "".join(out)


def zh_integer(n: int) -> str:
    if n == 0:
        return "零"
    if n < 0:
        return "负" + zh_integer(-n)
    groups: list[int] = []
    while n:
        groups.append(n % 10000)
        n //= 10000
    if len(groups) > len(ZH_GROUP_UNITS):
        return "".join(ZH_DIGIT_NAMES[d] for d in str(n))
    parts: list[str] = []
    need_zero = False
    for idx in range(len(groups) - 1, -1, -1):
        g = groups[idx]
        if g == 0:
            if parts:
                need_zero = True
            continue
        if parts and (need_zero or g < 1000):
            parts.append("零")
        word = _zh_group(g)
        if parts and word.startswith("十"):
            word = "一" + word
        parts.append(word + ZH_GROUP_UNITS[idx])
        need_zero = False
    return "".join(parts)


def en_integer(n: int) -> str:
    if n < 0:
        return "minus " + en_integer(-n)
    if n == 0:
        return "zero"
    if n > 10 ** 18:
        return " ".join(EN_ONES[int(d)] for d in str(n))
    groups: list[int] = []
    while n:
        groups.append(n % 1000)
        n //= 1000
    if len(groups) > len(EN_GROUP_UNITS):
        return " ".join(EN_ONES[int(d)] for d in str(n))

    def under_1000(v: int) -> str:
        parts: list[str] = []
        if v >= 100:
            parts.append(EN_ONES[v // 100] + " hundred")
            v %= 100
        if v >= 20:
            t, r = divmod(v, 10)
            word = EN_TENS[t]
            if r:
                word += "-" + EN_ONES[r]
            parts.append(word)
        elif v:
            parts.append(EN_ONES[v])
        return " ".join(parts)

    out: list[str] = []
    for idx in range(len(groups) - 1, -1, -1):
        g = groups[idx]
        if not g:
            continue
        word = under_1000(g)
        if EN_GROUP_UNITS[idx]:
            word += " " + EN_GROUP_UNITS[idx]
        out.append(word)
    return " ".join(out)


def _zh_digits(s: str) -> str:
    return "".join(ZH_DIGIT_NAMES.get(ch, ch) for ch in s)


def _en_digits(s: str) -> str:
    return " ".join(EN_ONES[int(ch)] for ch in s)


def _zh_number(raw: str, digit_by_digit: bool) -> str:
    raw = raw.replace(",", "")
    neg = raw.startswith("-")
    if neg:
        raw = raw[1:]
    if "." in raw:
        ip, fp = raw.split(".", 1)
        ip = ip or "0"
        value = (("负" if neg else "") + zh_integer(int(ip)) + "点" +
                 _zh_digits(fp))
        return value
    if digit_by_digit or (len(raw) > 1 and raw.startswith("0")):
        return ("负" if neg else "") + _zh_digits(raw)
    return ("负" if neg else "") + zh_integer(int(raw))


def _en_number(raw: str, digit_by_digit: bool) -> str:
    raw = raw.replace(",", "")
    neg = raw.startswith("-")
    if neg:
        raw = raw[1:]
    if "." in raw:
        ip, fp = raw.split(".", 1)
        ip = ip or "0"
        words = (("minus " if neg else "") + en_integer(int(ip)) +
                 " point " + " ".join(EN_ONES[int(ch)] for ch in fp))
        return words
    if digit_by_digit or (len(raw) > 1 and raw.startswith("0")):
        return ("minus " if neg else "") + _en_digits(raw)
    return ("minus " if neg else "") + en_integer(int(raw))


# A numeric expression, optionally with thousands separators and a decimal.
_NUMBER_RE = re.compile(
    r"(?<!\d)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d)")


def _has_unit_after(text: str, pos: int, lang: str) -> bool:
    rest = text[pos:]
    if lang == "zh":
        stripped = rest.lstrip()
        return bool(stripped) and stripped[0] in ZH_MEASURE
    m = re.match(r"\s*([A-Za-z]+)", rest)
    return bool(m and m.group(1).lower() in EN_MEASURE)


def _needs_digit_reading(raw: str, text: str, pos: int, lang: str) -> bool:
    digits = raw.replace(",", "").split(".", 1)[0]
    if digits.startswith("0") and len(digits) > 1:
        return True
    # Chinese years are read digit by digit: 2026年 -> 二零二六年.
    if (lang == "zh" and len(digits) == 4
            and text[pos:].lstrip().startswith("年")):
        return True
    # Ten or more unbroken digits are almost always an ID/phone/order code.
    if len(digits) >= 10 and not _has_unit_after(text, pos, lang):
        return True
    return False


def normalize_numbers(text: str, lang: str) -> str:
    """Replace Arabic numeral expressions with spoken words."""
    if not text:
        return text
    text = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    lang = "zh" if lang == "zh" else "en"
    out: list[str] = []
    last = 0
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(0)
        start, end = m.span()
        out.append(text[last:start])
        # Percent: consume the sign too.
        pct = re.match(r"\s*[%％]", text[end:])
        digit_read = _needs_digit_reading(raw, text, end, lang)
        if lang == "zh":
            spoken = (("百分之" if pct else "") +
                      _zh_number(raw, digit_read))
        else:
            spoken = (_en_number(raw, digit_read) +
                      (" percent" if pct else ""))
        if pct:
            end = pct.end() + end
        # Keep Latin/CJK boundaries readable: GPT4 -> GPT four / GPT4 -> GPT四
        prev = text[start - 1] if start else ""
        nxt = text[end] if end < len(text) else ""
        if lang == "en":
            if prev and prev.isascii() and prev.isalpha():
                spoken = " " + spoken
            if nxt and nxt.isascii() and nxt.isalpha():
                spoken += " "
        elif nxt and nxt.isascii() and nxt.isalpha():
            spoken += " "
        out.append(spoken)
        last = end
    out.append(text[last:])
    return "".join(out)
