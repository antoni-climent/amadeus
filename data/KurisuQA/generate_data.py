import csv 
import lmstudio as lms

with open('VNKurisuDialogues.csv', 'r') as file:
    with open('synthetic_data_v4.csv', 'w', newline='') as output_file:

        with lms.Client() as client:
            model = client.llm.model("deepseek-r1-distill-qwen-14b") #"qwen/qwen3.5-9b"

            writer = csv.writer(output_file)
            writer.writerow(['user', 'assistant'])  # Write header row
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            reader_list = list(reader)
            reader_length = len(reader_list)
            for n, row in enumerate(reader_list):
                message = f"""Write a sentence that another person could have said to receive this response sentence: '{row[1]}'. Only write the sentence, nothing else.
Example: 
Given this response sentence: I'm going to the convenience store to buy some pudding. 
Expected response: Where are you going?" 

Example 2:
Given this response sentence: Yes, I am. 
Expected response: Are you busy right now?

Example 3:
Given this response sentence: Because I was getting bored. 
Expected response: Why did you leave the party?"""



                generated_text = model.respond({"messages": [{"role": "user", "content": message}]})
                try:
                    generated_text = generated_text.content.split('<final_response>')[1].split('</final_response>')[0].strip()
                except:
                    generated_text = generated_text.content
                print(f"[{n}/{reader_length}] {generated_text}")
                writer.writerow([generated_text, row[1]])  # Write the user input and generated text to the CSV
                # Flush so that data is written incrementally
                output_file.flush()
                # Remove the last user message to avoid context buildup