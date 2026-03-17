import json

# Function to read a .jsonl file and process the data
def process_jsonl(file_path):
    valid_data = []
    video_data = {}
    
    # Read the .jsonl file
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            video_id = entry.get('id')
            #print(video_id)
            #print(entry)
            # Group entries by video_id
            if video_id not in video_data:
                video_data[video_id] = {'ori_qs': None, 'sub_qs': None, 'caption': None}
            
            # Store entries based on type
            if entry['type'] == 0:  # ori_qs
                video_data[video_id]['ori_qs'] = entry['ori_qs']
            elif entry['type'] == 1:  # sub_qs
                video_data[video_id]['sub_qs'] = entry['sub_qs']
            elif entry['type'] == 2:  # caption
                video_data[video_id]['caption'] = entry['caption']
    #print(video_data)
    # Now filter the videos where ori_qs is correct and sub_qs is incorrect
    for video_id, data in video_data.items():
        if data['ori_qs'] and data['sub_qs']:
            # Check if ori_qs is correct and sub_qs is wrong
            if data['ori_qs']['ans'] == data['ori_qs']['pred_response'] and data['sub_qs']['ans'] != data['sub_qs']['pred_response']:
                # Create change data format and save
                change_data = {
                    "aspect": "change",
                    "src_dataset": data['ori_qs']['src_dataset'],
                    "video_name": data['ori_qs']['video_name'],
                    "original_qa": {
                        "qs": data['ori_qs']['prompt'],
                        "choice": {
                            "A": data['ori_qs']['prompt'].split('Choices: ')[1].split(',')[0].strip(),
                            "B": data['ori_qs']['prompt'].split('Choices: ')[1].split(',')[1].strip(),
                            "C": data['ori_qs']['prompt'].split('Choices: ')[1].split(',')[2].strip(),
                        },
                        "ans": data['ori_qs']['ans']
                    },
                    "sub_qas": {
                        "sub_qa1": {
                            "qs": data['sub_qs']['prompt'],
                            "choice": {
                                "A": data['sub_qs']['prompt'].split('Choices: ')[1].split(',')[0].strip(),
                                "B": data['sub_qs']['prompt'].split('Choices: ')[1].split(',')[1].strip(),
                            },
                            "ans": data['sub_qs']['ans']
                        }
                    },
                    "caption": data['caption']['caption']
                }
                valid_data.append(change_data)

    # Save the valid data to a new JSONL file
    output_file_path = 'valid_video_data.jsonl'
    with open(output_file_path, 'w') as output_file:
        for entry in valid_data:
            json.dump(entry, output_file)
            output_file.write('\n')  # Write each entry on a new line

    print(f"Valid video data saved to '{output_file_path}'")
    print(f"Total valid entries: {len(valid_data)}")



# Call the function with the path to your .jsonl file
process_jsonl('/root/autodl-tmp/results/VILA1.5-3b/answer_pace.jsonl')
