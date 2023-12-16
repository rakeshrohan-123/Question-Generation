from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = AutoTokenizer.from_pretrained("kaejo98/bart-base_question_generation")
model = AutoModelForSeq2SeqLM.from_pretrained("kaejo98/bart-base_question_generation")

def run_model(input_string, generator_args, num_questions=10):
    input_string = "generate questions: " + input_string + " </s>"
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    
    # Remove 'max_length' from generator_args
    max_length = generator_args.pop("max_length", None)
    
    res = model.generate(input_ids, num_return_sequences=num_questions, max_length=max_length, **generator_args)
    
    output = tokenizer.batch_decode(res, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output

generator_args = {
    "num_beams": 10,
    "length_penalty": 1.5,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
}

@app.post("/generate_questions/")
async def generate_questions(request: Request):
    try:
        data = await request.json()
        text = data.get("text")
        if text:
            questions = run_model(text, generator_args, num_questions=10)
            return JSONResponse(content={"questions": questions})
        else:
            raise HTTPException(status_code=400, detail="Invalid input, 'text' field is required")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
