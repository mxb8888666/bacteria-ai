# -*- coding: utf-8 -*-
"""
CLSI 药敏判读标准 —— 抑菌圈直径断点表（单位：mm）

判读规则（K-B 纸片扩散法）：
  抑菌圈直径 ≥ S 断点  ->  敏感(S)
  抑菌圈直径 ≤ R 断点  ->  耐药(R)
  介于两者之间          ->  中介(I)

注意：这里收录的是【简化示例值】，用于演示判读逻辑。
真实临床必须查最新版 CLSI M100 标准文件——复试时强调这点很重要：
"标准是权威数据来源，代码只是把规则自动化"，说明你懂临床规范的严肃性。
"""

import json
import os

# 断点表统一放在 config.json：新增菌种/抗生素只改配置文件，不用改代码
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def _load_breakpoints():
    """读 config.json 的断点表，转成 {(菌种, 抗生素): {"S":.., "R":..}} 扁平字典。"""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    flat = {}
    for bac, abs_map in cfg["breakpoints"].items():
        for ab, b in abs_map.items():
            flat[(bac, ab)] = b
    return flat


BREAKPOINTS = _load_breakpoints()


def judge(bacteria, antibiotic, zone_mm):
    """根据抑菌圈直径(mm)判读 敏感(S)/中介(I)/耐药(R)。"""
    key = (bacteria, antibiotic)
    if key not in BREAKPOINTS:
        return "未知(标准未收录)"
    b = BREAKPOINTS[key]
    if b["S"] is not None and zone_mm >= b["S"]:
        return "敏感(S)"
    if b["R"] is not None and zone_mm <= b["R"]:
        return "耐药(R)"
    return "中介(I)"
