import json
import re
from pathlib import Path

def extract_value(field):
    """Extract value from either string or nested dict structure"""
    if isinstance(field, str):
        return field
    elif isinstance(field, dict) and "text" in field:
        text_list = field["text"]
        if isinstance(text_list, list) and len(text_list) > 0:
            return text_list[-1]  # Return the last element
        elif isinstance(text_list, str):
            return text_list
    return None

def _norm(s):
    if not isinstance(s, str):
        return ""
    return " ".join(s.split()).strip().lower()

def filter_qa_pairs(annotation_path):
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_key_re = re.compile(r'^(q|a|relevancy|correct|evidence|type)(\d+)$')
    cleaned = []
    kept_norm = set()
    global_question_counter = 0

    for rec in data:
        cleaned_rec = {k: v for k, v in rec.items() if not qa_key_re.match(k) and k != "qa_pairs"}
        indices = sorted({m.group(2) for k in rec.keys() if (m := qa_key_re.match(k))}, key=lambda x: int(x))
        valid_qa_pairs = []
        for idx in indices:
            q = rec.get(f"q{idx}")
            a = rec.get(f"a{idx}")
            relevancy = rec.get(f"relevancy{idx}")
            correct = rec.get(f"correct{idx}")
            evidence = rec.get(f"evidence{idx}")
            if isinstance(evidence, dict) and "choices" in evidence:
                evidence = evidence["choices"]
            type_val = rec.get(f"type{idx}")
            
            if relevancy == "relevant" and correct == "correct":
                global_question_counter += 1
                
                # Extract the actual question and answer values
                extracted_question = extract_value(q)
                extracted_answer = extract_value(a)
                
                # Skip if we can't extract valid question or answer
                if not extracted_question or not extracted_answer:
                    continue
                
                original_qa = None
                # Look for matching QA pair in qa_pairs array
                for qa in rec.get("qa_pairs", []) or []:
                    qa_question = extract_value(qa.get("question"))
                    if qa_question == extracted_question:
                        original_qa = {
                            "question": qa_question,
                            "answer": extract_value(qa.get("answer", qa.get("formatted_answer"))),
                            "evidence": evidence,
                            "type": type_val,
                            "question_id": f"q{global_question_counter}",
                        }
                        break
                
                if original_qa:
                    valid_qa_pairs.append(original_qa)
                else:
                    # Create new QA pair
                    qa_item = {
                        "question_id": f"q{global_question_counter}",
                        "question": extracted_question,
                        "answer": extracted_answer,
                        "evidence": evidence,
                        "type": type_val,
                    }
                    valid_qa_pairs.append(qa_item)
                    
                # Always preserve the original structure for valid QA pairs
                cleaned_rec[f"q{idx}"] = q
                cleaned_rec[f"a{idx}"] = a
                cleaned_rec[f"relevancy{idx}"] = relevancy
                cleaned_rec[f"correct{idx}"] = correct
                cleaned_rec[f"evidence{idx}"] = evidence
                cleaned_rec[f"type{idx}"] = type_val
                kept_norm.add(_norm(q))
        
        # Only add record if it has valid QA pairs
        if valid_qa_pairs:
            cleaned_rec["qa_pairs"] = valid_qa_pairs
            cleaned.append(cleaned_rec)

    ann_path_obj = Path(annotation_path)
    cleaned_annotation_path = f"{ann_path_obj.parent}/{ann_path_obj.stem}_filtered{ann_path_obj.suffix}" if ann_path_obj.suffix else f"{ann_path_obj.name}_filtered"
    with open(cleaned_annotation_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2)

    total_valid = sum(len(rec.get("qa_pairs", [])) for rec in cleaned)
    print(f"annotation cleaned: {cleaned_annotation_path} records: {len(data)} kept QA pairs: {total_valid}")

    return cleaned
