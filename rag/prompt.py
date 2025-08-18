GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
PROMPTS["DEFAULT_ENTITY_TYPES_MEDICINE"] = ["Disease", "Symptom", "Drug", "Pathogen", "Gene", "Protein", "DNA", "RNA",
                                            "Chemical", "Vaccine", "Tissue", "Toxin", "Procedure", "Equipment",
                                            "Person", "Location", "Organization", "Microorganism", "Animal",
                                            "Quantity", "Event"]

PROMPTS["question_decomposition_reqd"] = "There was a structural error in the last output. Please regenerate it."

PROMPTS["question_decomposition"] = """You are an expert in solving complex problems though decompositions. You need to follow the following steps to break down complex questions into simple points for retrieval.

Step 1. Question decomposition and response: Decompose the problem into multiple short sub-points. Or extract and summarize key sub-points, such as detailed symptoms, physical condition, dietary habits, etc
Include the following information:
   - sub_point: Short but complete sub-point, do not omit the subject or use pronouns, using words or phrases. Requirement: declarative and concisely.
   - sub_point_response: Only provide information according to the current sub point. Requirement: concisely. Not paying attention to the original problems. Do not make anything up. If unsure or unknown, return UNKNOWN.
Format each sub points and sub points response as ("sub"{tuple_delimiter}<sub_point>{tuple_delimiter}<sub_point_response>)

Step 2. Final response: Provide response to the original problem based on each decomposition points.
Respond the following information:
   - final_response: Respond the original problems based on the sub points. Do not make anything up. If unsure, unknown or no response required, return UNKNOW.
Format main-point and final response as ("response"{tuple_delimiter}<final_response>)

Step 3. Keyword extraction: List the high-level and low-level keywords in the problem.
Include the following information:
   - high-level keywords: focus on overarching concepts or themes.
   - low-level keywords: focus on specific entities, details, or concrete terms.
Format high-level keywords as ("high"{tuple_delimiter}<high_level_keywords>{tuple_delimiter}<high_level_keywords>)
Format low-level keywords as ("lower"{tuple_delimiter}<low_level_keywords>{tuple_delimiter}<low_level_keywords>)

Step 4. Return the output as a single list of all the information resolved in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

Note: Output strictly in the format of the example.

##########
-Examples-
##########
{examples}

##########
-Real Data-
##########
Question: {input_text}
##########
Output:
"""

PROMPTS["question_decomposition_examples"] = ["""Example 1:
Question: How do changes in microbial communities affect the development of diseases such as obesity and diabetes?
##########
("sub"{tuple_delimiter}"Microbial community changes impact on obesity development"{tuple_delimiter}"Alter gut microbiota composition and function, influencing metabolism and inflammation."){record_delimiter}
("sub"{tuple_delimiter}"Microbial community changes impact on diabetes development"{tuple_delimiter}"Modulate insulin sensitivity and glucose homeostasis through gut microbiota."){record_delimiter}
("response"{tuple_delimiter}"Changes in microbial communities can affect the development of diseases such as obesity and diabetes by altering gut microbiota composition and function, influencing metabolism and inflammation, and modulating insulin sensitivity and glucose homeostasis."){record_delimiter}
("high"{tuple_delimiter}"Microbial communities"{tuple_delimiter}"Disease development"{tuple_delimiter}"Obesity"{tuple_delimiter}"Diabetes"){record_delimiter}
("lower"{tuple_delimiter}"Gut microbiota"{tuple_delimiter}"Metabolism"{tuple_delimiter}"Inflammation"{tuple_delimiter}"Insulin sensitivity"{tuple_delimiter}"Glucose homeostasis"){record_delimiter}
##########""", """Example 2:
Question: Early safety outcome following transcatheter aortic valve implantation: is the amount of contrast media used a matter of concern?
##########
("sub"{tuple_delimiter}"Transcatheter aortic valve implantation procedure"{tuple_delimiter}"Minimally invasive procedure to replace aortic valve."){record_delimiter}
("sub"{tuple_delimiter}"Contrast media usage"{tuple_delimiter}"Used for imaging during procedure."){record_delimiter}
("sub"{tuple_delimiter}"Safety outcomes related to contrast media"{tuple_delimiter}"Potential risk of contrast-induced nephropathy."){record_delimiter}
("response"{tuple_delimiter}"The amount of contrast media used in transcatheter aortic valve implantation is a matter of concern due to potential safety outcomes such as contrast-induced nephropathy."){record_delimiter}
("high"{tuple_delimiter}"Transcatheter aortic valve implantation"{tuple_delimiter}"Safety outcomes"){record_delimiter}
("lower"{tuple_delimiter}"Contrast media"{tuple_delimiter}"Nephropathy"){record_delimiter}
##########""", """Example 3:
Question: A 60-year-old man seeks evaluation at a medical office due to leg pain while walking. He says the pain starts in his buttocks and extends to his thighs and down to his calves. Previously, the pain resolved with rest, but the pain now persists in his feet, even during rest. His past medical history is significant for diabetes mellitus, hypertension, and cigarette smoking. The vital signs are within normal limits. The physical examination shows an atrophied leg with bilateral loss of hair. Which of the following is the most likely cause of this patient’s condition?
##########
("sub"{tuple_delimiter}"Walking leg pain"{tuple_delimiter}"UNKNOWN"){record_delimiter}
("sub"{tuple_delimiter}"Persistent foot pain at rest"{tuple_delimiter}"UNKNOWN"){record_delimiter}
("sub"{tuple_delimiter}"Atrophied leg"{tuple_delimiter}"UNKNOWN"){record_delimiter}
("sub"{tuple_delimiter}"Bilateral hair loss"{tuple_delimiter}"UNKNOWN"){record_delimiter}
("response"{tuple_delimiter}"Based on the patient's symptoms of leg pain while walking, persistent foot pain at rest, atrophied leg, bilateral hair loss, and medical history of diabetes mellitus, hypertension, and cigarette smoking, the most likely cause of the patient's condition is narrowing and calcification of vessels."){record_delimiter}
("high"{tuple_delimiter}"Medical diagnosis"{tuple_delimiter}"Symptom evaluation"{tuple_delimiter}"Risk factors"){record_delimiter}
("lower"{tuple_delimiter}"Leg pain"{tuple_delimiter}"Foot pain"{tuple_delimiter}"Atrophied leg"{tuple_delimiter}"Loss of hair"){record_delimiter}
##########"""]

PROMPTS["question_decomposition_question"] = """You are an expert in solving complex problems though decompositions. You need to follow the following steps to break down complex questions into simple points for retrieval.

Step 1. Question decomposition and answer: Decompose the problem into the main-point and multiple short sub points.
Include the following information:
   - sub_question: Short but complete sub question. Requirement: concisely.
   - sub_question_answer: Answer only to the current sub question. Requirement: concisely. Not paying attention to the original problems. Do not make anything up. If unsure or unknown, return UNKNOWN.
Format each sub questions and sub questions answer as ("sub"{tuple_delimiter}<sub_point>{tuple_delimiter}<sub_point_answer>)

Step 2. Final response: Provide answers to the original problem based on each decomposition points.
Answer the following information:
   - final_response: Response the original problems based on the sub answer. Do not make anything up. If unsure or unknown, return UNKNOWN.
Format main-point and final answer as ("response"{tuple_delimiter}<final_answer>)

Step 3. Return the output as a single list of all the information resolved in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

##########
-Examples-
##########
{examples}

##########
-Real Data-
##########
Question: {input_text}
##########
Output:
"""

PROMPTS["question_decomposition_question_examples"] = ["""Example of knowing the answer:
Question: How did Shakespeare's works impact the development of English literature, and which of his plays is considered his masterpiece?
##########
("sub"{tuple_delimiter}How did Shakespeare's works impact the development of English literature?{tuple_delimiter}Shakespeare's works enriched the English language with new words and phrases, established many literary genres and dramatic structures, and had a profound influence on the themes and styles of later English literature.){record_delimiter}
("sub"{tuple_delimiter}Which of Shakespeare's plays is considered his masterpiece?{tuple_delimiter}Hamlet is widely regarded as one of Shakespeare's masterpieces.){record_delimiter}
("response"{tuple_delimiter}Shakespeare's works impacted the development of English literature by enriching the language, establishing literary genres and structures, and influencing themes and styles. Hamlet is considered one of his masterpieces.){record_delimiter}
##########""", """Example of not knowing the answer:
Question: How did Shakespeare's works impact the development of English literature, and which of his plays is considered his masterpiece?
##########
("sub"{tuple_delimiter}How did Shakespeare's works impact the development of English literature?{tuple_delimiter}UNKNOWN){record_delimiter}
("sub"{tuple_delimiter}Which of Shakespeare's plays is considered his masterpiece?{tuple_delimiter}UNKNOWN){record_delimiter}
("response"{tuple_delimiter}UNKNOWN){record_delimiter}
##########"""]

PROMPTS["text_summary"] = """You are an expert in document summarization in the fields of medicine and biology. You need to summarize the following text into a concise {language} sentence with no more than 20 words.

Supplement the summary into the following JSON and only return the complete JSON:
{{
    "summary": ""
}}

##########
-Examples-
##########
{examples}

##########
-Real Data-
##########
Text: {input_text}
##########
Output:
"""

PROMPTS["text_summary_examples"] = ["""Example 1:
Text: CONCLUSIONS: In this trial involving patients with HER2-low metastatic breast cancer, trastuzumab deruxtecan resulted in significantly longer progression-free and overall survival than the physician\u0027s choice of chemotherapy. (Funded by Daiichi Sankyo and AstraZeneca; DESTINY-Breast04 ClinicalTrials.gov number, NCT03734029.).
##########
Output:
{{
    "summary": "Trastuzumab deruxtecan improves survival in HER2-low metastatic breast cancer vs chemotherapy."
}}
##########""", """Example 2:
Text: Anemia resulting from iron and erythropoietin deficiencies is a common complication of advanced chronic kidney disease (CKD). This article covers major advances in our understanding of anemia in patients with CKD, including newly discovered regulatory molecules, such as hepcidin, to innovative intravenous iron therapies. The use of erythropoiesis-stimulating agents (ESA) in the treatment of anemia has undergone seismic shift in the past 3 years as a result of adverse outcomes associated with targeting higher hemoglobin levels with these agents. Potential mechanisms for adverse outcomes, such as higher mortality, are discussed. Despite the disappointing experience with ESAs, there is a tremendous interest in other novel agents to treat anemia in CKD. Lastly, while awaiting updated guidelines, the authors outline their recommendations on how to best manage patients who are anemic and have CKD.
##########
Output:
{{
    "summary": "Chronic kidney disease anemia management advances include hepcidin, intravenous iron, erythropoiesis-stimulating agents risks, and new therapies."
}}
##########"""]

PROMPTS["entity_extraction_reie"] = "There was a structural error in the last output. Please regenerate it."

PROMPTS["entity_extraction"] = """You are a medical and biological expert at extracting information in structured formats to build a knowledge graph. You need to follow the following steps to extract entities and relations.

Step 1. Identify specific entities related to medicine and biology in text, not conceptual vocabulary. Extract based on entity types in Entity_types. For all expressions referring to the same entity in the article, only the most specific version of the entity is retained to ensure that the entity is not duplicated.
For each identified entity, extract the following information:
   - entity_name: Entity name, capitalized.
   - entity_type: Entity type, one word.
   - entity_description: Comprehensive description of the entity's attributes and activities, focusing on its relevance in the fields of medicine and biology.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

Step 2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are related to each other.
For each pair of related entities, extract the following information:
   - source_entity: Must be the entity_name identified in step 1.
   - target_entity: Must be the entity_name identified in step 1.
   - relation_description: Comprehensive explanation between the source_entity and the target_entity, focusing on their relevance in medicine and biology.
   - relation_keywords: one or more high-level key words that summarize the overarching nature of the relation, focusing on concepts or themes rather than specific details.
Format each relation as ("relation"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relation_description>{tuple_delimiter}<relation_keywords>)

Step 3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

Step 4. Return the output as a single list of all the entities and relations identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

Note: Output strictly in the format of the example.

##########
-Examples-
##########
{examples}

##########
-Real Data-
##########
Entity_types: {entity_types}
Text: {input_text}
##########
Output:
"""

PROMPTS["entity_extraction_examples"] = ["""Example 1:
Entity_types: {entity_types}
Text: "In 1870, Hughlings Jackson, the eminent British neurologist, postulated that seizures were due to “an excessive and disorderly discharge of cerebral nervous tissue on muscles.” The discharge may result in an almost instantaneous loss of consciousness, alteration of perception or impairment of psychic function, convulsive movements, disturbance of sensation, or some combination thereof."
##########
("entity"{tuple_delimiter}"Year 1870"{tuple_delimiter}"Date"{tuple_delimiter}"The year when Hughlings Jackson postulated the cause of seizures"){record_delimiter}
("entity"{tuple_delimiter}"Hughlings Jackson"{tuple_delimiter}"Person"{tuple_delimiter}"A British neurologist known for his studies on epilepsy and other neurological disorders, who postulated the cause of seizures in 1870"){record_delimiter}
("entity"{tuple_delimiter}"Seizures"{tuple_delimiter}"Disease"{tuple_delimiter}"A neurological condition characterized by sudden, uncontrolled electrical disturbances in the brain, which may cause changes in behavior, movements, sensations, and levels of consciousness."){record_delimiter}
("entity"{tuple_delimiter}"Cerebral Nervous Tissue"{tuple_delimiter}"Tissue"{tuple_delimiter}"Refers to the brain tissue involved in the generation and transmission of electrical signals that can lead to seizures when excessively and disorderly discharged."){record_delimiter}
("entity"{tuple_delimiter}"Muscles"{tuple_delimiter}"Tissue"{tuple_delimiter}"Muscular tissues that can be affected by the discharge from cerebral nervous tissue, potentially resulting in convulsive movements during seizures."){record_delimiter}
("entity"{tuple_delimiter}"Consciousness Loss"{tuple_delimiter}"Symptom"{tuple_delimiter}"A sudden and almost instantaneous loss of awareness and responsiveness that can occur as a result of seizures."){record_delimiter}
("entity"{tuple_delimiter}"Alteration of Perception"{tuple_delimiter}"Symptom"{tuple_delimiter}"Changes in sensory perception, such as visual or auditory disturbances, that may occur during seizures."){record_delimiter}
("entity"{tuple_delimiter}"Psychic Function Impairment"{tuple_delimiter}"Symptom"{tuple_delimiter}"Impairment of mental functions, such as thinking, memory, or emotional regulation, associated with seizures."){record_delimiter}
("entity"{tuple_delimiter}"Convulsive Movements"{tuple_delimiter}"Symptom"{tuple_delimiter}"Involuntary, rapid, and rhythmic muscle contractions that can occur during seizures."){record_delimiter}
("entity"{tuple_delimiter}"Sensation Disturbance"{tuple_delimiter}"Symptom"{tuple_delimiter}"Abnormal sensations, such as tingling, numbness, or pain, that may be experienced during seizures."){record_delimiter}
("relation"{tuple_delimiter}"Year 1870"{tuple_delimiter}"Hughlings Jackson"{tuple_delimiter}"The year when Hughlings Jackson made his postulation about seizures"{tuple_delimiter}"time"){record_delimiter}
("relation"{tuple_delimiter}"Hughlings Jackson"{tuple_delimiter}"Seizures"{tuple_delimiter}"Hughlings Jackson postulated the cause of seizures in 1870"{tuple_delimiter}"research"){record_delimiter}
("relation"{tuple_delimiter}"Seizures"{tuple_delimiter}"Cerebral Nervous Tissue"{tuple_delimiter}"Seizures are caused by excessive and disorderly discharge from cerebral nervous tissue."{tuple_delimiter}"causation"){record_delimiter}
("relation"{tuple_delimiter}"Seizures"{tuple_delimiter}"Muscles"{tuple_delimiter}"Seizures can result in convulsive movements due to the effect on muscles."{tuple_delimiter}"effect"){record_delimiter}
("relation"{tuple_delimiter}"Seizures"{tuple_delimiter}"Consciousness Loss"{tuple_delimiter}"Seizures may cause an almost instantaneous loss of consciousness."{tuple_delimiter}"symptom"){record_delimiter}
("relation"{tuple_delimiter}"Seizures"{tuple_delimiter}"Alteration of Perception"{tuple_delimiter}"Seizures can lead to alterations in perception."{tuple_delimiter}"symptom"){record_delimiter}
("relation"{tuple_delimiter}"Seizures"{tuple_delimiter}"Psychic Function Impairment"{tuple_delimiter}"Seizures may result in impairment of psychic functions."{tuple_delimiter}"symptom"){record_delimiter}
("relation"{tuple_delimiter}"Seizures"{tuple_delimiter}"Convulsive Movements"{tuple_delimiter}"Seizures can cause convulsive movements."{tuple_delimiter}"symptom"){record_delimiter}
("relation"{tuple_delimiter}"Seizures"{tuple_delimiter}"Sensation Disturbance"{tuple_delimiter}"Seizures may lead to disturbances of sensation."{tuple_delimiter}"symptom"){record_delimiter}
("content_keywords"{tuple_delimiter}"Seizures"{tuple_delimiter}"Symptoms of Seizures"){tuple_delimiter}
##########""", """Example 2:
Entity_types: {entity_types}
Text: "Bacteria of the intestinal microbiota (see p. 372) can deconjugate (remove glycine and taurine) bile salts. They can also dehydroxylate carbon 7, producing secondary bile salts such as deoxycholic acid from cholic acid and lithocholic acid from chenodeoxycholic acid."
##########
("entity"{tuple_delimiter}"Intestinal Bacteria"{tuple_delimiter}"Microorganism"{tuple_delimiter}"Intestinal bacteria present in the intestinal microbiota that can deconjugate bile salts and dehydroxylate carbon 7 to produce secondary bile salts."){record_delimiter}
("entity"{tuple_delimiter}"Glycine"{tuple_delimiter}"Chemical"{tuple_delimiter}"An amino acid that can be removed from bile salts by intestinal bacteria."){record_delimiter}
("entity"{tuple_delimiter}"Taurine"{tuple_delimiter}"Chemical"{tuple_delimiter}"An amino acid that can be removed from bile salts by intestinal bacteria."){record_delimiter}
("entity"{tuple_delimiter}"Bile Salts"{tuple_delimiter}"Chemical"{tuple_delimiter}"Compounds that can be deconjugated by intestinal bacteria."){record_delimiter}
("entity"{tuple_delimiter}"Carbon 7"{tuple_delimiter}"Chemical"{tuple_delimiter}"A specific carbon atom in bile salts that can be dehydroxylated by intestinal bacteria."){record_delimiter}
("entity"{tuple_delimiter}"Deoxycholic Acid"{tuple_delimiter}"Chemical"{tuple_delimiter}"A secondary bile salt produced from cholic acid by intestinal bacteria."){record_delimiter}
("entity"{tuple_delimiter}"Cholic Acid"{tuple_delimiter}"Chemical"{tuple_delimiter}"A primary bile salt that can be converted into deoxycholic acid by intestinal bacteria."){record_delimiter}
("entity"{tuple_delimiter}"Lithocholic Acid"{tuple_delimiter}"Chemical"{tuple_delimiter}"A secondary bile salt produced from chenodeoxycholic acid by intestinal bacteria."){record_delimiter}
("entity"{tuple_delimiter}"Chenodeoxycholic Acid"{tuple_delimiter}"Chemical"{tuple_delimiter}"A primary bile salt that can be converted into lithocholic acid by intestinal bacteria."){record_delimiter}
("relation"{tuple_delimiter}"Intestinal Bacteria"{tuple_delimiter}"Bile Salts"{tuple_delimiter}"Intestinal bacteria can deconjugate bile salts."{tuple_delimiter}"interaction"){record_delimiter}
("relation"{tuple_delimiter}"Intestinal Bacteria"{tuple_delimiter}"Carbon 7"{tuple_delimiter}"Intestinal bacteria can dehydroxylate carbon 7 in bile salts."{tuple_delimiter}"dehydroxylation"){record_delimiter}
("relation"{tuple_delimiter}"Intestinal Bacteria"{tuple_delimiter}"Deoxycholic Acid"{tuple_delimiter}"Intestinal bacteria produce deoxycholic acid from cholic acid."{tuple_delimiter}"production"){record_delimiter}
("relation"{tuple_delimiter}"Intestinal Bacteria"{tuple_delimiter}"Cholic Acid"{tuple_delimiter}"Intestinal bacteria convert cholic acid into deoxycholic acid."{tuple_delimiter}"metabolism"){record_delimiter}
("relation"{tuple_delimiter}"Intestinal Bacteria"{tuple_delimiter}"Lithocholic Acid"{tuple_delimiter}"Intestinal bacteria produce lithocholic acid from chenodeoxycholic acid."{tuple_delimiter}"production"){record_delimiter}
("relation"{tuple_delimiter}"Intestinal Bacteria"{tuple_delimiter}"Chenodeoxycholic Acid"{tuple_delimiter}"Intestinal bacteria convert chenodeoxycholic acid into lithocholic acid."{tuple_delimiter}"metabolism"){record_delimiter}
("relation"{tuple_delimiter}"Intestinal Bacteria"{tuple_delimiter}"Glycine"{tuple_delimiter}"Intestinal bacteria can remove glycine from bile salts."{tuple_delimiter}"deconjugation"){record_delimiter}
("relation"{tuple_delimiter}"Intestinal Bacteria"{tuple_delimiter}"Taurine"{tuple_delimiter}"Intestinal bacteria can remove taurine from bile salts."{tuple_delimiter}"deconjugation"){record_delimiter}
("content_keywords"{tuple_delimiter}"Intestinal Bacteria"{tuple_delimiter}"Microbiota"{tuple_delimiter}"Bile Salts"{tuple_delimiter}"Metabolism"{tuple_delimiter}"Secondary Bile Salts"){record_delimiter}
##########"""]

PROMPTS["entiti_continue_extraction"] = """MANY entities and relations may have been lost in the last extraction. If they are missing, add them below using the same format. The last extraction result is as follows:
"""

PROMPTS["entiti_if_loop_extraction"] = """It appears some entities may have still been missed. Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["summarize_entity_descriptions"] = """You are an expert in analyzing and summarizing information in the fields of medicine and biology, responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

##########
-Data-
Entities: {entity_name}
Description List: {description_list}
##########
Output:
"""
generate_error = """The format of the generated content is incorrect. Please regenerate or modify it. This is the {num}-th generation. {error} The last generated result is as follows: """

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

# todo 根据数据集情况，需要在这里进行修改，用于匹配对应的响应类型。
PROMPTS["rag_response"] = """You are a medical and biological expert responding to questions about data in the tables provided.

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

-Target response length and format-
{response_type}

-Data tables-
{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["rag_response_bioasq_ideal"] = """You are a medical and biological expert responding to questions about data in the tables provided.

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
Only return a string to summarize the answer. Simplify as much as possible, within 200 words. If you don't know the answer, just say so. Do not make anything up.

-Data tables-
{context_data}
"""

PROMPTS["rag_response_bioasq_yesno"] = """You are a medical and biological expert responding to questions about data in the tables provided.

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
You must answer only with lowercase 'yes' or 'no' even if you are not sure about the answer.

-Data tables-
{context_data}
"""

PROMPTS["rag_response_bioasq_factoid"] = """You are a medical and biological expert responding to questions about data in the tables provided.

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
Answer this question by returning a JSON string array called 'entities of entity names, numbers, or similar short expressions that are an answer to the question, ordered by decreasing confidence. The array should contain at max 5 elements but can contain less. If you don't know any answer return an empty array.
Return only this array, containing a maximum of 20 elements, it must not contain phrases and **must be valid JSON**. Example: {{\"entities\": [\"entity1\", \"entity2\"]}}"

-Data tables-
{context_data}
"""

PROMPTS["rag_response_bioasq_list"] = """You are a medical and biological expert responding to questions about data in the tables provided.

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
Answer this question by only returning a JSON string array called 'entities of entity names, numbers, or similar short expressions that are an answer to the question (e.g., the most common symptoms of a disease). The returned array will have to contain no more than 20 entries of no more than 100 characters each. If you don't know any answer return an empty array.
Return only this array, it must not contain phrases and **must be valid JSON**. Example: {{\"entities\": [\"entity1\", \"entity2\"]}}"

-Data tables-
{context_data}
"""

PROMPTS["rag_response_pubmedqa"] = """You are a medical and biological expert responding to questions about data in the tables provided.

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
Including the following information:
 - answer_long: 1-5 sentences, containing only the response to the question.
 - answer_decision: The finalized answer, a single-word ("yes", "no", or "maybe").

Supplement the response into the following JSON and only return the complete JSON:
{{
    "answer_long": "",
    "answer_decision": ""
}}

---Example 1---
question: "Is the adjunctive use of light-activated disinfection ( LAD ) with FotoSan ineffective in the treatment of peri-implantitis : 1-year results from a multicentre pragmatic randomised controlled trial?"
{{
    "answer_long": "Adjunctive use of LAD therapy (FotoSan) with mechanical cleaning of implants affected by peri-implantitis did not improve any clinical outcomes when compared to mechanical cleaning alone up to 1 year after treatment.",
    "answer_decision": "yes"
}}

---Example 2---
question: "Is a mutation in the 5,10-methylenetetrahydrofolate reductase gene associated with preeclampsia in women of southeast Mexico?"
{{
    "answer_long": "Our results suggest that C677T-MTHFR polymorphism is not an associated risk factor for developing preeclampsia in southeast Mexico. Results also confirmed high prevalence of C677T mutation in Mexico.",
    "answer_decision": "no"
}}

---Data tables---
{context_data}
"""

PROMPTS["naive_rag_response"] = """You are a helpful assistant responding to questions about documents provided.

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

-Target response length and format-
{response_type}

-Documents-
{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["similarity_check"] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate the following two points and provide a similarity score between 0 and 1 directly:
1. Whether these two questions are semantically similar
2. Whether the answer to Question 2 can be used to answer Question 1
Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""

# customization prompt
PROMPTS["entity_type_custom"] = ["Disease", "Symptom", "Protein", "Gene", "Drug", "Pathway", "Human", "Animal", "Cell",
                                 "Preparation", "Pharmacodynamics", "Pharmacokinetics"]

PROMPTS["entity_extraction_custom"] = """You are a medical and biological expert at extracting information in structured formats to build a knowledge graph. You need to follow the following steps to extract entities and relations.

Step 1. Identify specific entities related to medicine and biology in text, not conceptual vocabulary. Extract based on entity types in Entity_types. For all expressions referring to the same entity in the article, only the most specific version of the entity is retained to ensure that the entity is not duplicated.
For each identified entity, extract the following information:
   - entity_name: Entity name, capitalized.
   - entity_type: Entity type, one word.
   - entity_description: Comprehensive description of the entity's attributes and activities, focusing on its relevance in the fields of medicine and biology.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

Step 2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are related to each other.
For each pair of related entities, extract the following information:
   - source_entity: Must be the entity_name identified in step 1.
   - target_entity: Must be the entity_name identified in step 1.
   - relation_description: Comprehensive explanation between the source_entity and the target_entity, focusing on their relevance in medicine and biology.
   - relation_keywords: one high-level keyword that summarize the overarching nature of the relation, focusing on concepts or themes rather than specific details.
Format each relation as ("relation"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relation_description>{tuple_delimiter}<relation_keywords>)

Step 3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

Step 4. Return the output as a single list of all the entities and relations identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

Note: Output strictly in the format of the example.

##########
-Examples-
##########
{examples}

##########
-Real Data-
##########
Entity_types: {entity_types}
Some entity type descriptions:
"Drug": Chemicals for treating diseases, nucleic acid drugs, vaccines, excluding toxic agent and insecticide.
"Human": Patients and healthy volunteers participated in clinical trials.
"Cell": Cell models for testing drug efficacy or safety.
"Preparation": Drug formulation or drug dosage form.
"Pharmacodynamics": Study of drug absorption, distribution, metabolism and excretion.
Text: {input_text}
##########
Output:
"""

PROMPTS["entity_extraction_examples_custom"] = ["""Example 1:
Entity_types: {entity_types}
Text: "In the military, combat wound infections can progress rapidly to life-threatening sepsis. The discovery of effective small-molecule drugs to prevent and/or treat sepsis is a priority. To identify potential sepsis drug candidates, we used an optimized larval zebrafish model of endotoxicity/sepsis to screen commercial libraries of drugs approved by the U.S. Food and Drug Administration (FDA) and other active pharmaceutical ingredients (APIs) known to affect pathways implicated in the initiation and progression of sepsis in humans (i.e., inflammation, mitochondrial dysfunction, coagulation, and apoptosis). We induced endotoxicity in 3- and 5-day post fertilization larval zebrafish (characterized by mortality and tail fin edema (vascular leakage)) by immersion exposure to 60 µg/mL Pseudomonas aeruginosa lipopolysaccharide (LPS) for 24 h, then screened for the rescue potential of 644 selected drugs at 10 µM through simultaneous exposure to LPS."
##########
("entity"{tuple_delimiter}"COMBAT WOUND INFECTIONS"{tuple_delimiter}"Disease"{tuple_delimiter}"Infections that occur in wounds sustained during combat, which can rapidly progress to life-threatening sepsis."){record_delimiter}
("entity"{tuple_delimiter}"SEPSIS"{tuple_delimiter}"Disease"{tuple_delimiter}"A life-threatening condition caused by the body's response to infection, leading to tissue damage, organ failure, and death."){record_delimiter}
("entity"{tuple_delimiter}"LARVAL ZEBRAFISH"{tuple_delimiter}"Animal"{tuple_delimiter}"A developmental stage of zebrafish used as a model organism to study endotoxicity and sepsis."){record_delimiter}
("relation"{tuple_delimiter}"COMBAT WOUND INFECTIONS"{tuple_delimiter}"SEPSIS"{tuple_delimiter}"Combat wound infections can rapidly progress to sepsis, a life-threatening condition."{tuple_delimiter}"disease progression"){record_delimiter}
("content_keywords"{tuple_delimiter}"sepsis, combat wound infections, zebrafish model, drug screening, endotoxicity")
##########""", """Example 2:
Entity_types: {entity_types}
Text: "We induced endotoxicity in 3- and 5-day post fertilization larval zebrafish (characterized by mortality and tail fin edema (vascular leakage)) by immersion exposure to 60 µg/mL Pseudomonas aeruginosa lipopolysaccharide (LPS) for 24 h, then screened for the rescue potential of 644 selected drugs at 10 µM through simultaneous exposure to LPS. After LPS exposure, we used a neurobehavioral assay (light-dark test) to further evaluate rescue from endotoxicity and to determine possible off-target drug side effects. We identified 29 drugs with > 60% rescue of tail edema and mortality. Three drugs (Ketanserin, Tegaserod, and Brexpiprazole) produced 100% rescue and did not differ from the controls in the light-dark test, suggesting a lack of off-target neurobehavioral effects. Further testing of these three drugs at a nearly 100% lethal concentration of Klebsiella pneumoniae LPS (45 µg/mL) showed 100% rescue from mortality and 88–100% mitigation against tail edema.The success of the three identified drugs in a zebrafish endotoxicity/sepsis model warrants further evaluation in mammalian sepsis models."
##########
("entity"{tuple_delimiter}"ENDOTOXICITY"{tuple_delimiter}"Disease"{tuple_delimiter}"A condition induced by exposure to bacterial lipopolysaccharides, characterized by systemic inflammation and organ dysfunction."){record_delimiter}
("entity"{tuple_delimiter}"LARVAL ZEBRAFISH"{tuple_delimiter}"Animal"{tuple_delimiter}"A developmental stage of zebrafish used as a model organism to study endotoxicity and sepsis."){record_delimiter}
("entity"{tuple_delimiter}"KETANSERIN"{tuple_delimiter}"Drug"{tuple_delimiter}"A serotonin antagonist identified as a potential rescue drug in zebrafish endotoxicity models, showing 100% rescue from mortality."){record_delimiter}
("entity"{tuple_delimiter}"TEGASEROD"{tuple_delimiter}"Drug"{tuple_delimiter}"A serotonin receptor agonist identified as a potential rescue drug in zebrafish endotoxicity models, showing 100% rescue from mortality."){record_delimiter}
("entity"{tuple_delimiter}"BREXPIPRAZOLE"{tuple_delimiter}"Drug"{tuple_delimiter}"An antipsychotic drug identified as a potential rescue drug in zebrafish endotoxicity models, showing 100% rescue from mortality."){record_delimiter}
("entity"{tuple_delimiter}"NEUROBEHAVIORAL ASSAY"{tuple_delimiter}"Pharmacodynamics"{tuple_delimiter}"A light-dark test used to evaluate rescue from endotoxicity and determine possible off-target drug side effects."){record_delimiter}
("relation"{tuple_delimiter}"LARVAL ZEBRAFISH"{tuple_delimiter}"ENDOTOXICITY"{tuple_delimiter}"Larval zebrafish are used as a model organism to study endotoxicity induced by bacterial lipopolysaccharides."{tuple_delimiter}"model organism"){record_delimiter}
("relation"{tuple_delimiter}"KETANSERIN"{tuple_delimiter}"ENDOTOXICITY"{tuple_delimiter}"Ketanserin showed 100% rescue from mortality in zebrafish endotoxicity models."{tuple_delimiter}"rescue"){record_delimiter}
("relation"{tuple_delimiter}"TEGASEROD"{tuple_delimiter}"ENDOTOXICITY"{tuple_delimiter}"Tegaserod showed 100% rescue from mortality in zebrafish endotoxicity models."{tuple_delimiter}"rescue"){record_delimiter}
("relation"{tuple_delimiter}"BREXPIPRAZOLE"{tuple_delimiter}"ENDOTOXICITY"{tuple_delimiter}"Brexpiprazole showed 100% rescue from mortality in zebrafish endotoxicity models."{tuple_delimiter}"rescue"){record_delimiter}
("relation"{tuple_delimiter}"NEUROBEHAVIORAL ASSAY"{tuple_delimiter}"ENDOTOXICITY"{tuple_delimiter}"A neurobehavioral assay (light-dark test) was used to evaluate rescue from endotoxicity and detect off-target drug effects."{tuple_delimiter}"drug effect test"){record_delimiter}
("content_keywords"{tuple_delimiter}"endotoxicity, zebrafish model, drug rescue, lipopolysaccharide induction, neurobehavioral assay")
##########"""]
