from dataclasses import dataclass


@dataclass(frozen=True)
class Criterion:
    name: str
    label_ja: str
    left: str
    right: str
    description: str


CRITERIA: dict[str, Criterion] = {
    "left_right": Criterion(
        name="left_right",
        label_ja="左派 ↔ 右派",
        left="左派",
        right="右派",
        description="改革・変革を志向する左派的な政治姿勢か、伝統・秩序・現状維持を重視する右派的な政治姿勢か",
    ),
    "topdown_bottomup": Criterion(
        name="topdown_bottomup",
        label_ja="トップダウン ↔ ボトムアップ",
        left="トップダウン",
        right="ボトムアップ",
        description="強いリーダーシップで上から引っ張るタイプか、合意形成を重視し下から積み上げるタイプか",
    ),
    "romanticist_realist": Criterion(
        name="romanticist_realist",
        label_ja="ロマンチスト ↔ リアリスト",
        left="ロマンチスト",
        right="リアリスト",
        description="理想や信念を追い求めるタイプか、現実的・実利的な判断を重視するタイプか",
    ),
    "dog_cat": Criterion(
        name="dog_cat",
        label_ja="犬っぽい ↔ 猫っぽい",
        left="犬っぽい",
        right="猫っぽい",
        description="忠実で人懐っこく集団行動を好む犬っぽい雰囲気か、独立心が強くマイペースな猫っぽい雰囲気か",
    ),
    "carnivore_herbivore": Criterion(
        name="carnivore_herbivore",
        label_ja="肉食系 ↔ 草食系",
        left="肉食系",
        right="草食系",
        description="積極的・攻撃的・野心的な肉食系タイプか、穏やか・控えめ・受動的な草食系タイプか",
    ),
    "inkya_youkya": Criterion(
        name="inkya_youkya",
        label_ja="陰キャ ↔ 陽キャ",
        left="陰キャ",
        right="陽キャ",
        description="内向的・暗い・ひとりが好きな雰囲気か、外向的・明るい・社交的な雰囲気か",
    ),
}

DEFAULT_CRITERION = "left_right"
