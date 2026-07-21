# 宪法规则引擎配方

## 规则结构

- name, severity, applies_when, must, explanation, fix
- 谓词: contains_regex, ends_with_regex, all_of, any_of, not_
- 原子: contains_regex, not_contains_regex, ends_with_regex, max_words

## 修正器

append_if_missing, prepend_if_missing, replace_regex

## 输出

违规列表 + 修订版 + diff
