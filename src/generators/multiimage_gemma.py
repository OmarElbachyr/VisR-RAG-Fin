import ollama


test_queries = {
    "What risk represents the largest percentage of risk-weighted assets at both the end of 2016 and September 2017?": [
        "vqa-ir-qa/data/indexed_pages/8b85b4aece2ed0404ff63cfb31f1e9ed_1.png",
        "vqa-ir-qa/data/indexed_pages/8b85b4aece2ed0404ff63cfb31f1e9ed_3.png",
        "vqa-ir-qa/data/indexed_pages/8b85b4aece2ed0404ff63cfb31f1e9ed_4.png"
    ],
    "What was the Tier 1 capital ratio in 2017?": [
        "vqa-ir-qa/data/indexed_pages/8b85b4aece2ed0404ff63cfb31f1e9ed_2.png",
        "vqa-ir-qa/data/indexed_pages/8b85b4aece2ed0404ff63cfb31f1e9ed_3.png",
        "vqa-ir-qa/data/indexed_pages/8b85b4aece2ed0404ff63cfb31f1e9ed_4.png"
    ],
}

for query, image_list in test_queries.items():
    response = ollama.chat(model='gemma3:4b', 
        messages=[{
            'role': 'user', 
            'content': query,
            'images': image_list,
        }],
        # options={"temperature":0.7}
        )

    print(response['message']['content'])
