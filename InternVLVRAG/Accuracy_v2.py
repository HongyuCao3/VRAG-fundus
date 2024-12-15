import pandas as pd

# Read evaluation data
df = pd.read_csv('evaluation.csv')

# Calculate modality accuracy
# CFP modality
cfp_gt = df[df['gt_ans'] == "This is a color fundus image."]
cfp_count_gt = cfp_gt.shape[0]
cfp_count_answer = cfp_gt['answer'].str.contains("color fundus", case=False).sum()

# FFA modality
ffa_gt = df[df['gt_ans'] == "This is a fundus fluorescein angiography (FFA) image."]
ffa_count_gt = ffa_gt.shape[0]
ffa_count_answer = ffa_gt['answer'].str.contains("fundus fluorescein angiography|FFA", case=False).sum()

# OCT modality
oct_gt = df[df['gt_ans'] == "This is an optical coherence tomography (OCT) image."]
oct_count_gt = oct_gt.shape[0]
oct_count_answer = oct_gt['answer'].str.contains("optical coherence tomography|OCT", case=False).sum()

# Calculate modality accuracy
cfp_accuracy = cfp_count_answer / cfp_count_gt
ffa_accuracy = ffa_count_answer / ffa_count_gt
oct_accuracy = oct_count_answer / oct_count_gt

average_modality_accuracy = (cfp_count_answer + ffa_count_answer + oct_count_answer) / (cfp_count_gt + ffa_count_gt + oct_count_gt)

# Output modality accuracy
print(f"CFP modality accuracy: {cfp_accuracy:.2%}")
print(f"FFA modality accuracy: {ffa_accuracy:.2%}")
print(f"OCT modality accuracy: {oct_accuracy:.2%}")
print(f"Average modality accuracy: {average_modality_accuracy:.2%}")

# Calculate eye accuracy
left_eye_gt = df[df['gt_ans'] == "Left eye."]
left_eye_count_gt = left_eye_gt.shape[0]
left_eye_count_answer = left_eye_gt['answer'].str.contains("left eye", case=False).sum()

right_eye_gt = df[df['gt_ans'] == "Right eye."]
right_eye_count_gt = right_eye_gt.shape[0]
right_eye_count_answer = right_eye_gt['answer'].str.contains("right eye", case=False).sum()

left_eye_accuracy = left_eye_count_answer / left_eye_count_gt
right_eye_accuracy = right_eye_count_answer / right_eye_count_gt

average_eye_accuracy = (left_eye_count_answer + right_eye_count_answer) / (left_eye_count_gt + right_eye_count_gt)

# Output eye accuracy
print(f"Left eye accuracy: {left_eye_accuracy:.2%}")
print(f"Right eye accuracy: {right_eye_accuracy:.2%}")
print(f"Average eye accuracy: {average_eye_accuracy:.2%}")

# Read mapping relationship
mapping_df = pd.read_excel('mapping relationship.xlsx')

# Create input to general diagnosis mapping dictionary
mapping_dict = dict(zip(mapping_df['input'], mapping_df['general diagnosis']))

# Extract diagnosis
df['diagnosis'] = df['gt_ans'].str.extract(r'The possible diagnosis of this image is (.+?)\.')[0]

# Calculate correct diagnosis count
correct_diagnosis_count = 0
diagnosis_count = 0

# Iterate through each row, check if answer contains general diagnosis or its reverse mapping in mapping relationship.xlsx
for index, row in df.iterrows():
    # Get corresponding general diagnosis
    general_diagnosis = mapping_dict.get(row['diagnosis'])

    if general_diagnosis is not None:
        diagnosis_count += 1
        # Collect all possible inputs
        possible_inputs = mapping_df[mapping_df['general diagnosis'] == general_diagnosis]['input'].tolist()

        # Check if answer contains general diagnosis or its corresponding inputs
        if general_diagnosis in row['answer'] or any(input_item in row['answer'] for input_item in possible_inputs):
            correct_diagnosis_count += 1

# Calculate diagnosis accuracy
diagnosis_accuracy = correct_diagnosis_count / diagnosis_count if diagnosis_count > 0 else 0
print(f"Diagnosis accuracy: {diagnosis_accuracy:.2%}")