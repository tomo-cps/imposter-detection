import json
from pathlib import Path
from typing import Dict, List, Callable, Any, Optional
import pandas as pd
import torch
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score
from models import model_selection
from prompts.imposter_detection_prompts import get_default_system_prompt, get_zero_shot_prompt, get_few_shot_prompt
from datetime import datetime
from enum import Enum, auto
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)

class PromptType(Enum):
    ZERO_SHOT = auto()
    FEW_SHOT = auto()

PROMPT_FUNC_MAPPING: Dict[PromptType, Callable[[str, str], str]] = {
    PromptType.ZERO_SHOT: get_zero_shot_prompt,
    PromptType.FEW_SHOT: get_few_shot_prompt
}

class ImposterDetectProcessor:
    def __init__(
        self, 
        model_name: str, 
        system_prompt: Optional[str] = None, 
        prompt_func: Optional[Callable[[str, str], str]] = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_new_tokens: int = 50,
        output_dir: str = "outputs"
    ) -> None:
        logger.info(f"Loading model '{model_name}'...")
        self.model, self.tokenizer = model_selection.load_model(model_name)
        logger.info("Model loading completed.")

        self.system_prompt = system_prompt if system_prompt is not None else get_default_system_prompt()
        self.prompt_func = prompt_func if prompt_func is not None else get_zero_shot_prompt

        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.output_dir = Path(output_dir)

    def process_file(self, input_path: Path, prompt_type: PromptType) -> None:
        logger.info(f"Starting processing file {input_path} with prompt type: {prompt_type.name}")
        try:
            df = self._load_data(input_path)
            logger.info("Data loaded and context generated.")
            processed_data = self._get_results(df)
            output_path = self._get_output_path(prompt_type, input_path)
            self._save_output(processed_data, output_path)
            self._display_result(df)
        except Exception as e:
            logger.error("An error occurred:")
            logger.exception(e)
            traceback.print_exc()
    
    def _load_data(self, input_path: Path) -> pd.DataFrame:
        df = pd.read_csv(input_path)
        df['context'] = df.groupby('context_label')['user_comment'].transform(lambda x: ' '.join(x))
        return df
    
    def _get_prompt(self, context: str, user_comment: str) -> torch.Tensor:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt_func(context, user_comment)},
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
    
    def _predict_label(self, context: str, user_comment: str) -> str:
        token_ids = self._get_prompt(context, user_comment)
        with torch.no_grad():
            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        output = self.tokenizer.decode(
            output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True
        )
        return output.strip()
    
    def _get_results(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        logger.info("Starting prediction with the model.")
        predicted_labels = []
        for _, row in df.iterrows():
            context = row['context']
            user_comment = row['user_comment']
            prediction = self._predict_label(context, user_comment)
            predicted_labels.append(True if prediction == "TRUE" else False)
        
        df['pred_label'] = predicted_labels
        accuracy = accuracy_score(df['label'], df['pred_label'])
        df['accuracy'] = f"Accuracy: {accuracy * 100:.2f}%"
        logger.info(f"Prediction completed. {df['accuracy'].iloc[0]}")
        output_data = df.to_dict(orient='records')
        return output_data
    
    def _get_output_path(self, prompt_type: PromptType, input_path: Path) -> Path:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_name = input_path.stem
        output_dir = self.output_dir / prompt_type.name.lower()
        output_path = output_dir / f"{current_time}_{data_name}.json"
        return output_path
    
    def _save_output(self, output_data, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Results saved to {output_path}")
        return output_path

    def _display_result(self, df: pd.DataFrame) -> None:
        from tabulate import tabulate  
        logger.info("Displaying the results.")

        accuracy = df['accuracy'].iloc[0]
        logger.info(f"Overall accuracy: {accuracy}")

        display_df = df[['context_label', 'user_comment', 'label', 'pred_label']]
        logger.info("\n" + tabulate(display_df, headers='keys', tablefmt='github', showindex=False))


def main() -> None:
    logger.info("Main process started.")
    model_name = "elyza/Llama-3-ELYZA-JP-8B"
    system_prompt = get_default_system_prompt()
    
    prompt_type = PromptType.FEW_SHOT  # ZERO_SHOT, FEW_SHOT
    prompt_func = PROMPT_FUNC_MAPPING[prompt_type]
    
    input_path = Path("data/sample.csv")
    
    processor = ImposterDetectProcessor(
        model_name,
        system_prompt=system_prompt,
        prompt_func=prompt_func
    )
    
    processor.process_file(input_path, prompt_type)
    logger.info("Main process finished.")

if __name__ == "__main__":
    main()
