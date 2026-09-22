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

BREAKPOINTS = {
    ("大肠杆菌", "头孢他啶"):   {"S": 21, "R": 17},
    ("大肠杆菌", "环丙沙星"):   {"S": 21, "R": 15},
    ("大肠杆菌", "阿米卡星"):   {"S": 17, "R": 14},
    ("大肠杆菌", "左氧氟沙星"): {"S": 19, "R": 13},
    ("金黄色葡萄球菌", "青霉素"):   {"S": 29, "R": 28},
    ("金黄色葡萄球菌", "万古霉素"): {"S": 17, "R": None},  # R=None 表示该药无耐药断点
}


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
