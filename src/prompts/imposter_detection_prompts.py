def get_default_system_prompt() -> str:
    """デフォルトのシステムプロンプトを返す関数"""
    return f"""
あなたは人狼ゲームのスペシャリストでルールについても熟知しています．
与えられた文脈と発言を元に，それが嘘か本当かを判断してください．
"""

def get_zero_shot_prompt(context: str, user_comment: str) -> str:
    """ゼロショットで推論するプロンプトを返す関数"""
    return f"""
以下の文脈からユーザの発言が嘘か本当か判断してください．

文脈: 
{context}\n

発言: 
{user_comment}\n

この発言は嘘ですか？\n
TRUE または FALSE で答えてください。
"""

def get_few_shot_prompt(context: str, user_comment: str) -> str:
    """ゼロショットで推論するプロンプトを返す関数"""
    return f"""
以下の文脈からユーザの発言が嘘か本当か判断してください．

この発言は嘘ですか？\n
TRUE または FALSE で回答してください，

例：
文脈: 
こんにちは 私は占い師です Aさんを占った結果，市民でした 私も占い師です．Aさんを占った結果，人狼でした Bさんは少し声が震えてて怪しいです
発言: 
私も占い師です．Aさんを占った結果，人狼でした

回答: 
TRUE

文脈: 
{context}\n

発言: 
{user_comment}\n

回答: 
"""

