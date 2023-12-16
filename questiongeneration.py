from transformers import T5ForConditionalGeneration, T5TokenizerFast

hfmodel = T5ForConditionalGeneration.from_pretrained("ThomasSimonini/t5-end2end-question-generation")
tokenizer = T5TokenizerFast.from_pretrained("ThomasSimonini/t5-end2end-question-generation")

text = "The abolition of feudal privileges by the National Constituent Assembly on 4 August 1789 and the Declaration \\nof the Rights of Man and of the Citizen (La Déclaration des Droits de l'Homme et du Citoyen), drafted by Lafayette \\nwith the help of Thomas Jefferson and adopted on 26 August, paved the way to a Constitutional Monarchy \\n(4 September 1791 – 21 September 1792). Despite these dramatic changes, life at the court continued, while the situation \\nin Paris was becoming critical because of bread shortages in September. On 5 October 1789, a crowd from Paris descended upon Versailles \\nand forced the royal family to move to the Tuileries Palace in Paris, where they lived under a form of house arrest under \\nthe watch of Lafayette's Garde Nationale, while the Comte de Provence and his wife were allowed to reside in the \\nPetit Luxembourg, where they remained until they went into exile on 20 June 1791."

def run_model(input_string, **generator_args):
    generator_args = {
        "max_length": 256,
        "num_beams": 4,
        "length_penalty": 1.5,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }
    input_string = "generate questions: " + input_string + " </s>"
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    
    questions = []
    num_questions_to_generate = 10  # Adjust this value to the desired number of questions
    
    for _ in range(num_questions_to_generate):
        res = hfmodel.generate(input_ids, **generator_args)
        output = tokenizer.decode(res[0], skip_special_tokens=True)
        questions.append(output)

    return questions

generated_questions = run_model(text)
print(generated_questions)
