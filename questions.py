from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with your frontend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

hfmodel = T5ForConditionalGeneration.from_pretrained("ThomasSimonini/t5-end2end-question-generation")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
num_questions=10
def run_model(input_string, generator_args):
    input_string = "generate questions: " + input_string + " </s>"
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = hfmodel.generate(input_ids,num_return_sequences=num_questions, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    output = [item.split("<extra_id_-1>") for item in output]
    return output

generator_args = {
    "max_length": 1000,
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
            questions = run_model(text, generator_args)
            return JSONResponse(content={"questions": questions})
        else:
            raise HTTPException(status_code=400, detail="Invalid input, 'text' field is required")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

