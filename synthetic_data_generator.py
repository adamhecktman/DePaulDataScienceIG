"""
How to Generate Synthetic Data with OpenAI's Language Model
Run the script: Open your terminal or command prompt and navigate to the directory containing the script and input file. Then, execute the following command:
python3 synthetic_data_generator.py --data_path input_data.xlsx --output_dir ./synthetic_output --example_c 50 --workers 5
Arguments Explanation:
--data_path: Path to your Excel input file (input_data.xlsx in this case).
--output_dir: Directory where the generated synthetic data will be saved (./synthetic_output here).
--example_c: Number of examples to generate per outcome (set to 50 in this example).
--workers: Number of parallel workers to use for processing (set to 5 here).


# required libraries
pip install openai pandas tqdm

# At the end you wil get a CSV file with the synthetic data


PLEASE REPLACE THE API KEY WITH YOURS
"""


import argparse
import json
import os
import time
import uuid

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

OPANAI_KEY = "XXX"
client = OpenAI(api_key=OPANAI_KEY)


PROMPT = """You are provided with examples and data related to a program. Your task is to produce one JSON object with exactly three top-level keys: "program", "outcome", and "outcome_id". Follow these guidelines:

1. **program**:
   - Write a short yet clear description of the program based on the given data.
   - Mention the relevant Impact Areas and Genome Information in a natural way with different synonyms but maintain the original meaning and context. Do not use Initiative or Program in the description.
   - Incorporate any key details from the data, but do not copy entire example text verbatim.
   - Do not include additional JSON or lists inside this field.
   - Be creative and original in your writing, naming of organizations, and program details.
   - The Organization name can be fictional and does not need to be consistent across examples. Can be the name of a place, item, fruit, animal, color, planet, vehicle, profession, plant, or mineral.
1. **Output Format**:
   - Return only one JSON object with this one field and nothing else.

6. **Examples**:
{examples}

7. **Data**:
{data}
""".format

# Function to invoke a language model (LLM) with a given prompt and return the response
def invoke_llm_model(prompt):
    # Define the model
    model = "gpt-4o"
    # Call the language model with a prompt and some parameters
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You make an program description with provided data and examples."},
            {"role": "user", "content": prompt},
        ],
        # Define the expected response format as a JSON schema
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "program": {"description": "Program description", "type": "string"},
                    },
                    "additionalProperties": False
                }
        }},
        # Adjust parameters for response generation
        temperature=0.3,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # Return the generated content from the model's response
    return response.choices[0].message.content

# Function to generate a prompt and send it to the LLM
def prompt_model(data, examples):
    # Create the full prompt using provided examples and data
    prompt = PROMPT(examples=examples, data=data)
    # Invoke the model with the generated prompt
    data = invoke_llm_model(prompt)
    # Attempt to parse the model's response into a JSON object
    try:
        generation = json.loads(data)
    except Exception as e:
        print(f"Cannot make json object from the data: {e}: {data}")
        return
    # Return the parsed JSON object if successful
    return generation

# Function to create examples from input data
def construct_examples(data, example_c=5):
    examples = []  # List to store generated examples
    cc = 0  # Counter to track the number of examples
    for i, row in data.iterrows():
        # Format a string for each example based on specific columns in the data
        example = f"Impact Areas: {row['impactarea']}\nGenome Information: {row['genome']} Outcome: {row['outcome']} Outcome ID: {row['outcomeid']}\n programdescription: {row['programdescription']}"
        examples.append(example)
        cc += 1
        if cc == example_c:  # Stop when the desired number of examples is reached
            break
    return examples

# Function to generate synthetic data and save it to a file
def generate_synthetic_data(data, examples, impact_area, genome, outcome, outcome_id, output_dir):
    # Use the prompt model to generate data
    generation = prompt_model(data, examples)
    if generation:
        try:
            # Create a unique filename for the generated data
            name = f"{generation['outcome']}_{str(uuid.uuid4())}.json"
            # Define the path to save the file
            pth = os.path.join(output_dir, name)
            # Add additional metadata to the generation
            generation['outcome'] = outcome
            generation['outcome_id'] = outcome_id
            generation['impact_area'] = impact_area
            generation['genome'] = genome
            # Save the generated data to a JSON file
            with open(pth, 'w') as f:
                json.dump(generation, f, indent=4, ensure_ascii=False)
            print(f"Generated: {name}")
        except Exception as e:
            print(f"Error saving file: {e}")



# Function to process the input data and generate synthetic examples
def process_data(data_path, output_dir, example_c=100, workers=10):
    """
    Process input data from an Excel file, generate synthetic examples,
    and save them to the specified output directory.
    """
    # Load the data from an Excel file
    df = pd.read_excel(data_path)
    # Remove duplicate rows based on the 'programdescription' column
    df = df.drop_duplicates(subset=['programdescription'])
    # Get the unique outcome IDs
    outcomes = df['outcomeid'].unique()
    # Use a thread pool to parallelize synthetic data generation
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []  # List to track future tasks
        for outcome in outcomes:
            # Filter the data for the current outcome ID
            outcome_df = df[df['outcomeid'] == outcome]
            # If there are fewer examples than needed, generate more
            if len(outcome_df) < example_c:
                print(f"Processing outcome: {outcome} because we have {len(outcome_df)} examples.")
                how_many = example_c - len(outcome_df)
                for i in range(0, how_many):
                    # Sample data to create examples
                    df_same_outcome = df[df['outcomeid'] == outcome]
                    if len(df_same_outcome) > 15:
                        examples_sample = df[df['outcomeid'] == outcome].sample(5)
                        examples = construct_examples(examples_sample)
                    else:
                        examples_sample = df.sample(15)
                        examples = construct_examples(examples_sample)
                    try:
                        # Format the input data for synthetic generation
                        data = f"{'Impact Areas: ' + outcome_df['impactarea'].values[0]}\n{'Genome Information: ' + outcome_df['genome'].values[0]} Outcome: {outcome_df['outcome'].values[0]} Outcome ID: {outcome_df['outcomeid'].values[0]}"
                        # Submit a task to the thread pool
                        futures.append(executor.submit(generate_synthetic_data, data, examples, outcome_df['impactarea'].values[0], outcome_df['genome'].values[0], outcome_df['outcome'].values[0], outcome_df['outcomeid'].values[0], output_dir))
                    except Exception as e:
                        print(f"Error generating synthetic data: {e}")
        print("Total tasks in queue: ", len(futures))
        # Wait for all tasks to complete and show progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating examples"):
            future.result()

def json_to_csv(input_dir, output_file):
    # Function to convert JSON files to a CSV file
    data = [] # List to store data from JSON files
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as f:
                json_data = json.load(f) # Load JSON data from file
                # Extract relevant fields and add them to the data list
                data.append({
                    'programdescription': json_data.get('program', ''),
                    'impactarea': json_data.get('impact_area', ''),
                    'genome': json_data.get('genome', ''),
                    'outcome': json_data.get('outcome', ''),
                    'outcomeid': json_data.get('outcome_id', '')
                })
    # Create a DataFrame from the collected data and save it to
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data to generate synthetic examples.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input data file (Excel format).')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the generated synthetic data.')
    parser.add_argument('--example_c', type=int, default=100,
                        help='Number of examples to generate per outcome (default: 100).')
    parser.add_argument('--workers', type=int, default=10, help='Number of workers to use for parallel processing')

    args = parser.parse_args()
    print(f"Processing data from: {args.data_path}")
    print(f"Saving synthetic data to: {args.output_dir}")
    print(f"Number of examples to generate per outcome: {args.example_c}")
    print(f"Number of workers: {args.workers}")
    process_data(args.data_path, args.output_dir, args.example_c, args.workers)

    # Convert the generated synthetic data to a CSV file
    # get parent directory of the output_dir
    parent_dir = os.path.dirname(args.output_dir)
    # make output file with date
    output_file = os.path.join(parent_dir, f"synthetic_data_{time.strftime('%Y%m%d')}.csv")
    json_to_csv(args.output_dir, output_file)
    print(f"Generated CSV file: {output_file}")
