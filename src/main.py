import json
from pathlib import Path
from typing import Dict, List, Callable
import pandas as pd
import torch
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score
from models import model_selection
from prompts.imposter_detection_prompts import get_default_system_prompt, get_zero_shot_prompt, get_few_shot_prompt
from datetime import datetime
from enum import Enum, auto

class PromptType(Enum):
    ZERO_SHOT = auto()
    FEW_SHOT = auto()
    # 必要に応じて他のプロンプトタイプを追加

PROMPT_FUNC_MAPPING = {
    PromptType.ZERO_SHOT: get_zero_shot_prompt,
    PromptType.FEW_SHOT: get_few_shot_prompt
}

def get_output_path(prompt_type: PromptType, input_path: Path) -> Path:
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_name = input_path.stem
    output_dir = Path("outputs") / prompt_type.name.lower()  # "outputs/zero_shot"など
    output_path = output_dir / f"{current_time}_{data_name}.json"
    return output_path


class ImposterDetectProcessor:
    def __init__(self, model_name: str, 
        system_prompt: str = None, 
        zero_shot_prompt_func: Callable[[str, str], str] = None):
        self.model, self.tokenizer = model_selection.load_model(model_name)
        self.system_prompt = system_prompt if system_prompt is not None else get_default_system_prompt()
        self.zero_shot_prompt_func = zero_shot_prompt_func if zero_shot_prompt_func is not None else get_zero_shot_prompt
        
    def process_file(self, input_path: Path, output_path: Path):
        try:
            df = self._load_data(input_path)
            output_data = self._get_results(df)
            output_path = self._save_output(output_data, output_path)
            print(f'{output_path}に保存されました')
            
        except Exception as e:
            print("エラーが発生しました:")
            print(f"エラーの詳細: {e}")
            traceback.print_exc()
    
    def _load_data(self, input_path):
        df = pd.read_csv(input_path)
        df['context'] = df.groupby('context_label')['user_comment'].transform(lambda x: ' '.join(x))
        return df
    
    def _get_prompt(self, context, user_comment): 
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.zero_shot_prompt_func(context, user_comment)},
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        token_ids = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        return token_ids
    
    def _predict_label(self, context, user_comment):
        token_ids = self._get_prompt(context, user_comment)
        with torch.no_grad():
            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                max_new_tokens=50,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
        output = self.tokenizer.decode(
            output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True
        )
        return output.strip()
    
    def _get_results(self, df):
        predicted_labels = []
        for i, row in df.iterrows():
            context = row['context']
            user_comment = row['user_comment']
            prediction = self._predict_label(context, user_comment)
            predicted_labels.append(True if prediction == "TRUE" else False)
            
        df['pred_label'] = predicted_labels
        
        accuracy = accuracy_score(df['label'], df['pred_label'])
        accuracy = f"正解率: {accuracy * 100:.2f}%"
        df['accuracy'] = accuracy
        output_data = df.to_dict(orient='records')
        
        return output_data
    
    def _save_output(self, output_data, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        return output_path

def main():
    model_name = "elyza/Llama-3-ELYZA-JP-8B"
    system_prompt = get_default_system_prompt()
    
    prompt_type = PromptType.FEW_SHOT
    prompt_func = PROMPT_FUNC_MAPPING[prompt_type]
    
    input_path = Path("data/sample.csv")
    output_path = get_output_path(prompt_type, input_path)
    
    processor = ImposterDetectProcessor(
        model_name,
        system_prompt=system_prompt,
        zero_shot_prompt_func=prompt_func
    )
    
    processor.process_file(input_path, output_path)

if __name__ == "__main__":
    main()
